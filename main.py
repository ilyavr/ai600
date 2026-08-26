import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ConnectionException
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import os 
import time

dir_name = os.getcwd()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CENTERED_IMG_DIR = os.path.join(DATA_DIR, "centered_images")
UNCENTERED_IMG_DIR = os.path.join(DATA_DIR, "uncentered_images")
LOG_FILE = os.path.join(DATA_DIR, 'ocr_log.txt')

for d in [DATA_DIR, CENTERED_IMG_DIR, UNCENTERED_IMG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ====================== LOGGER ==================
class LocalTimeFormatter(logging.Formatter):
    tz = ZoneInfo("Asia/Krasnoyarsk")

    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, self.tz)
        return dt.timetuple()
    
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, ct)
        else:
            return time.strftime("%Y-%m-%d %H:%M:%S", ct)

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
logging.Logger.success = success

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = LocalTimeFormatter('%(levelname)-10s | %(asctime)s.%(msecs)03d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False

# ===============================================================================================================================

model_class_path = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')

CONTROLLER_IP = "192.168.2.34"
CONTROLLER_PORT = 502
trigger_reg = 0

SNAPSHOT_URL = "http://192.168.3.115/ISAPI/Streaming/channels/101/picture"
CAM_LOGIN = "admin"
CAM_PASSWORD = "kZx_vN8!"

ZONE_X1, ZONE_Y1 = 960, 250  
ZONE_X2, ZONE_Y2 = 1130, 890  

isCentered = False

model_class = YOLO(model_class_path)

conn = ModbusTcpClient(CONTROLLER_IP, port=CONTROLLER_PORT, timeout=2, retries=2)
coordDict = {}
coordMass = []
trigger_already_handled = False 

def writeCenteringResult():
    global isCentered
    try:
        if isCentered:
            conn.write_register(address=1, value=1)
            logger.success("Отцентрировано правильно, записываю 1 в D1")
        else:
            conn.write_register(address=2, value=1)
            logger.success("Не отцентрировано, записываю 1 в D2")

        response = conn.read_holding_registers(address=0, count=3)
        if not response.isError():
            registers = response.registers
            logger.info(f"Значения регистров: {registers}")
            for idx, val in enumerate(registers):
                logger.info(f"Регистр {idx}: {val}")
    except Exception as e:
        logger.error(f"Ошибка при записи/чтении Modbus: {e}")
    
    isCentered = False

def ensure_modbus_connected():
    if not conn.connect():
        if not conn.connect():
            time.sleep(2)
            return False
    return True

while True:
    if not ensure_modbus_connected():
        continue

    try:
        trigger = conn.read_holding_registers(address=trigger_reg, count=1)
        if trigger.isError():
            time.sleep(0.5)
            continue
        trigger_val = trigger.registers[0]
        
    except ConnectionException:
        conn.close()
        time.sleep(1)
        continue

    if trigger_val == 0:
        time.sleep(0.1)
        continue

    if trigger_already_handled:
        time.sleep(0.1)
        continue

    try:
        firstScanner = conn.read_holding_registers(address=20, count=1)
        secondScanner = conn.read_holding_registers(address=30, count=1)
        reg = firstScanner.registers + secondScanner.registers
    except Exception as e:
        logger.error(f"Ошибка чтения сканеров: {e}")
        time.sleep(1)
        continue

    logger.info("\n Сработал триггер")
    logger.info(f"Данные сканеров: {reg}")
    trigger_already_handled = True 

    time.sleep(0.1)

    try:
        response = requests.get(SNAPSHOT_URL, auth=HTTPDigestAuth(CAM_LOGIN, CAM_PASSWORD), timeout=2)
        if response.status_code != 200:
            logger.error(f"Ошибка камеры: HTTP {response.status_code}")
        else:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                object_found = False
                FILE_TO_SAVE = None  
                attempt = 0
                MAX_ABSOLUTE_RETRIES = 100 

                while not object_found and attempt < MAX_ABSOLUTE_RETRIES:
                    attempt += 1
                    draw_frame = frame.copy()
                    cv2.rectangle(draw_frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)
                    if attempt == 1:
                        debug_path = os.path.join(UNCENTERED_IMG_DIR, f"DEBUG_RAW_INPUT.jpg")
                        cv2.imwrite(debug_path, draw_frame)
                        logger.info(f"ОТЛАДКА: Сырой кадр от камеры сохранен в {debug_path}. Размер: {draw_frame.shape}")
                    results_class = model_class(draw_frame, verbose=False, conf=0.01)

                    if len(results_class[0].boxes) > 0:
                        logger.info(f"ОТЛАДКА: Найдено боксов вообще: {len(results_class[0].boxes)}")
                        for idx, b in enumerate(results_class[0].boxes):
                            dbg_cls = int(b.cls.cpu().numpy()[0])
                            dbg_conf = float(b.conf.cpu().numpy()[0])
                            logger.info(f"  -> Бокс {idx}: Класс={model_class.names[dbg_cls]}, Уверенность={dbg_conf:.3f}")
                        # =======================================================

                        high_conf_boxes = [b for b in results_class[0].boxes if float(b.conf.cpu().numpy()[0]) >= 0.50]
                        
                        if len(high_conf_boxes) > 0:
                            object_found = True
                            box = high_conf_boxes[0].xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        
                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        cv2.circle(draw_frame, (x_center, y_center), 5, (0, 0, 255), -1)

                        model_name = "Unknown"
                        prob_val = 0.0

                        class_id = int(results_class[0].boxes.cls.cpu().numpy()[0])
                        model_name = model_class.names[class_id]
                        prob_val = float(results_class[0].boxes.conf.cpu().numpy()[0])

                        scanner_match = (str(int(reg[0])) in model_name) or (str(int(reg[1])) in model_name)
                        
                        if ZONE_X1 <= x_center <= ZONE_X2 and ZONE_Y1 <= y_center <= ZONE_Y2 and scanner_match:
                            isCentered = True
                            status_text = "STATUS: OK"
                            text_color = (0, 255, 0) 
                            logger.info(f"Результат: {model_name} (Conf: {prob_val:.2f}) отцентрован (попытка {attempt})")
                            coordDict.setdefault(model_name, []).append([x_center, y_center])
                            logger.info(f'{coordDict}')
                            photoTimestamp = int(time.time())
                            FILE_TO_SAVE = os.path.join(CENTERED_IMG_DIR, f"{photoTimestamp}.jpg")
                        elif scanner_match == False:
                            status_text = "STATUS: WRONG SCANNER OR CV MODEL"
                            text_color = (0, 0, 255) 
                            logger.info(f"Результат: {model_name} (Conf: {prob_val:.2f}) НЕ совпадает штрих-код (попытка {attempt})")
                            photoTimestamp = int(time.time())
                            FILE_TO_SAVE = os.path.join(UNCENTERED_IMG_DIR, f"{photoTimestamp}.jpg")
                        else:
                            isCentered = False
                            status_text = "STATUS: OFFSET"
                            text_color = (0, 0, 255) 
                            logger.info(f"Результат: {model_name} (Conf: {prob_val:.2f}) смещен (попытка {attempt})")
                            photoTimestamp = int(time.time())
                            FILE_TO_SAVE = os.path.join(UNCENTERED_IMG_DIR, f"{photoTimestamp}.jpg")
                        
                        writeCenteringResult()

                        cv2.putText(draw_frame, f"Scanners: {reg}", (x1, y1 -100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(draw_frame, f"Recognized: {model_name}", (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(draw_frame, f"Conf: {prob_val:.2f}", (x1, y1 - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(draw_frame, status_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

                        frame = draw_frame
                    else:
                        logger.warning(f"Попытка {attempt}/{MAX_ABSOLUTE_RETRIES}: Объект не найден на фото. Повторяем...")
                        time.sleep(0.05)

                if not object_found:
                    logger.critical(f"Объект не найден после {MAX_ABSOLUTE_RETRIES} попыток!")
                    photoTimestamp = int(time.time())
                    FILE_TO_SAVE = os.path.join(UNCENTERED_IMG_DIR, f"CRITICAL_FAIL_{photoTimestamp}.jpg")
                    cv2.imwrite(FILE_TO_SAVE, frame)
                    logger.error(f"Ошибка сохранена в {FILE_TO_SAVE}")

                if object_found and FILE_TO_SAVE is not None:
                    cv2.imwrite(FILE_TO_SAVE, frame)
                    logger.info(f"Фото сохранено в {FILE_TO_SAVE}")
            else:
                logger.error("Камера вернула битый файл.")
    except Exception as e:
        logger.error(f"Ошибка при съемке: {e}")

    logger.info("Ждем сброса триггера в 0...")
    while True:
        try:
            trig_check = conn.read_holding_registers(address=trigger_reg, count=1)
            if not trig_check.isError() and trig_check.registers[0] == 0:
                break
        except: pass
        time.sleep(0.1)

    logger.info("Ждем установки триггера в 1...")
    while True:
        try:
            trig_check = conn.read_holding_registers(address=trigger_reg, count=1)
            if not trig_check.isError() and trig_check.registers[0] == 1:
                break
        except: pass
        time.sleep(0.1)

    trigger_already_handled = False