import os

LABEL_FOLDERS = [
    r"C:\Users\Volkov-iv\Desktop\ppucase_center\modelsByClasses\labels\train",
    r"C:\Users\Volkov-iv\Desktop\ppucase_center\modelsByClasses\labels\val"
]


COMBINATION_JSON = {
    "0":[235,1146,1646,250],
    "1":[315],
    "2":[634,134,6134],
    "3":[4292,2501,542,2502,6142],
    "4":[6133,4312,133,1133,633],
    "5":[6143],
    "6":[920,820,320,1920],
    "7":[1147,147,1647,6047],
    "8":[132,632,6132],
    "9":[960,860,360,1960],
    "10":[125,4342,631,6131,131],
    "11":[940,340,840,1940],
    "12":[6130,6127,294,627,3342],
    "13":[649,149,6129,1149,6149],
    "14":[980,880,380,1980],
    "15":[135, 139],
    "16":[6041,6141],
    "17":[646,146,4231,2505,6046]
}

mapping = {}
for new_class, old_models in COMBINATION_JSON.items():
    for old_model in old_models:
        mapping[str(old_model)] = str(new_class)

files_changed = 0
labels_replaced = 0
unmapped_classes = set()

for folder in LABEL_FOLDERS:
    if not os.path.exists(folder): 
        continue
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            new_lines = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        old_class = parts[0]
                        if old_class in mapping:
                            parts[0] = mapping[old_class]
                            labels_replaced += 1
                        else:
                            unmapped_classes.add(old_class)
                        new_lines.append(" ".join(parts))
            
            with open(filepath, 'w') as f:
                f.write("\n".join(new_lines))
            files_changed += 1

print(f"Обработано файлов: {files_changed}")
print(f"Заменено строк (ббоксов): {labels_replaced}")

if unmapped_classes:
    print(f"\nВ файлах найдены модели, которых НЕТ в ai600_combination.json: {unmapped_classes}")

print("-" * 40)
print("nc: 18")
print("names:")
for new_class in range(18):
    old_models = COMBINATION_JSON[str(new_class)]
    print(f"  {new_class}: {old_models}")
print("-" * 40)