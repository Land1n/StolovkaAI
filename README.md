# РЕАЛИЗАЦИЯ ТРЕКИНГА ЛЮДЕЙ В ОЧЕРЕДИ
# Версия 2
##  Заставка
https://github.com/Land1n/StolovkaAI/assets/165151853/750914c0-db79-4d76-a540-8c97e88b8b0c
## Набор файлов 

* dataset_yolov8 - набор данных для обучения
* runs/detect/train - обучение нейросети
* video_test - видео для теста
* StolovkaAI_v1n.pt - первая версия нашей нейросети
* botsort.yaml - трекинг ( https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)

## Результаты 
![confusion_matrix_normalized](https://github.com/Land1n/StolovkaAI/assets/165151853/ec785316-455a-4bb1-a026-db319446b3da)
![P_curve](https://github.com/Land1n/StolovkaAI/assets/165151853/e0de3fa7-b7fc-42c6-8f3c-2547c114a3b0)
![labels](https://github.com/Land1n/StolovkaAI/assets/165151853/de3acab8-7ac5-462a-85ec-9e51d97b3c27)

# Версия 1 
##  Заставка
https://github.com/Land1n/StolovkaAI/assets/165151853/a7a8d665-3274-4ddf-b001-e186dd13f477
## Набор файлов 

* main.py - основной файл с кодом
* mask.png - маска для изображения, чтобы нейросеть провела поиск объектов в нужной области
* requirements.txt - необходимые зависимости
* yolov8n.pt - обученная нейросеть 
* video.mp4 - склеенный видео файл

# Установка:

1. Склонируйте к себе этот репозиторий
```
git clone https://github.com/Land1n/StolovkaAI.git
```
2. Перейдите с помощью команды cd в созданную папку StolovkaAI
```
cd StolovkaAI
```
3. Загрузите все необходимые библиотеки: 
```
pip install -r requirements.txt
```
4. Запустите main.py
