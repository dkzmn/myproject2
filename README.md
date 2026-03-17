# Домашнее задание 4: Fine-tuning MusicGen

## Установка окружения

```bash
git clone https://github.com/dkzmn/HSE-DLA-HW4.git
cd HSE-DLA-HW4

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r audiocraft/requirements.txt
python -m pip install gdown datasets ollama
python -m pip install av --no-deps
python -m pip install -e audiocraft
dvc pull data/wav
dvc pull dataset.csv
```

## Сбор данных MusicCaps
Данные уже в DVC, этот пункт можно пропустить
```bash
python scripts/download_musiccaps.py --output-dir data --skip-existing
```

## Обогащение метаданных с помощью LLM
Данные уже в DVC, этот пункт можно пропустить
```bash
ollama pull llama3.1:8b
ollama run llama3.1:8b
python scripts/create_json.py --models llama3.1:8b --skip-existing
```

После этого проверял данные в блокноте [notebooks/check_dataset.ipynb](notebooks/check_dataset.ipynb)

## Настройка конфигов и запуск обучения

```bash
python scripts/prepare_audiocraft_manifests.py --wav-dir data/wav --manifests-dir data/manifests
```

### Запуск fine-tuning (small)
```bash
python -m audiocraft.train \
  solver=musicgen/musicgen_hw4_32khz_finetune \
  continue_from=//pretrained/facebook/musicgen-small
```

Для обучения на Yandex DataSphere использовал ноутбук [notebooks/notebook_for_yandex.ipynb](notebooks/notebook_for_yandex.ipynb)


## Оценка качества генерации

WAV файлы, сгенерированные с помощью лучшей дообученной модели находятся тут: [generated_wav](generated_wav)

### Генерация 5 тестовых промптов

Бинарники модели (small 5 эпох, lr=e-4) тут: [https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing](https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing)

Для скачивания можно выполнить команду:
```bash
python scripts/download_model.py
```

```bash
python scripts/generate_test_tracks.py \
  --model-path checkpoints/best_model_small \
  --prompts-dir prompts \
  --output-dir generated_wav \
  --duration 12 \
  --device cuda
```


---

## Формат сдачи
1. Ссылка на GitHub-репозиторий с вашим кодом (скрипты парсинга, модифицированный AudioCraft, скрипты инференса).
https://github.com/dkzmn/HSE-DLA-HW4.git

2. Ссылка на веса обученной модели (HuggingFace / Google Drive).
https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing

3. Папка с 5 сгенерированными `.wav` файлами, названными `prompt_1.wav` ... `prompt_5.wav`.
[generated_wav](generated_wav)

4. Краткий отчет (Markdown/PDF) с описанием:
   * С какими трудностями столкнулись?
   Долгое обучение. Почти сразу получил более менее приемлимый результат на small модели, после 5 эпох 
   с LR=0.0001 (эксперимент best_small в CometML) Лучшего результата не смог добиться ни дообученим этого чекпоинта с меньшим LR, ни другими попытками
   Также пытался medium модель на GPU A100, но очень долго, не успел, хотя графики там, кажется, хорошие
   (эксперимент best_medium в CometML)

   * Какую LLM использовали для парсинга и какой системный промпт сработал лучше всего?
   ollama llama3.1:8b - бесплатно и без лимитов, кажется со своей задачей справилась
   промпт тут: [prompts/prompt_for_llm_1.txt](prompts/prompt_for_llm_1.txt) - его составить помогла также LLM

   * Какие гиперпараметры обучения (learning rate, batch size, steps) вы использовали?
   LR экспериментировал от e-6 до e-3, batch size не менял

   * Приложите логи обучения (ссылку на WandB/CometML), чтобы можно было оценить процесс обучения.
   [https://www.comet.com/dkzmn/hw4-musicgen/view/new/panels](https://www.comet.com/dkzmn/hw4-musicgen/view/new/panels) лучшие эксперименты запинены.


