# gd-levelgen

> English version: [README.md](../README.md). Это исходное описание проекта на русском. В публичный репозиторий
> не входят данные (`data/`, `replay/`), исследовательские материалы из `re/` (включая `re/physics_spec.md`)
> и Geode SDK (`geode/`).

Генерация уровней Geometry Dash. Цель — качество уровня featured.

## Данные

Краулер обходит список featured-уровней (`getGJLevels21`, type=6, ~36k уровней) и скачивает каждый
(`downloadGJLevel22`). Один запрос раз в ~2 с, при rate limit / Cloudflare — растущие паузы до 30 мин.
Платформеры (length=5) по умолчанию не скачиваются, только метаданные.

```bash
./crawl start            # в фоне, продолжает с места остановки
./crawl status           # прогресс (не трогает БД)
./crawl log              # tail -f лога
./crawl stop             # мягкая остановка
./crawl start --include-platformer   # докачать и платформеры
```

Всё лежит в `data/levels.sqlite`, таблица `levels`:
метаданные (stars, epic 0..3, feature_score, length, song_id, ...), `list_meta`/`dl_meta` — сырые поля
ответа сервера в JSON, `data` — сжатая строка уровня.

```python
import sqlite3
from gdgen.levelstring import decompress, parse_level
db = sqlite3.connect("data/levels.sqlite")
blob, = db.execute("SELECT data FROM levels WHERE status='downloaded' LIMIT 1").fetchone()
header, objects = parse_level(decompress(blob))   # objects: [{'1': id, '2': x, '3': y, ...}, ...]
```

## Токены

`gdgen/tokens.py` переводит уровень в токены и обратно без видимых потерь (позиции с точностью до 0.25,
углы до 0.1°, редакторские ключи отбрасываются; `scripts/token_report.py` проверяет round-trip):

```
<BOS> <HEADER> H:kA4 V:1 ... <COLORS> <CHAN> CP:6 V:1000 ... <OBJECTS>
SEG:0 O:1 X:60 Y:0 YF:60 FS:64.67.96 A:21=1004 G:12 ... SEG:1 ... <EOS>
```

`SEG:s` — сегмент в 10 блоков, `X/Y/YF` — позиция, дальше атрибуты: частые наборы флагов (`FS:`), группы (`G:`),
частые пары ключ=значение одним токеном (`A:k=v`), остальное — `K:k` + значение. Словарь
(`gdgen/data/vocab_v1.json`, < 65536 токенов) подбирается `scripts/fit_vocab.py`. В среднем ~9 токенов на объект
геймплея, ~11 на декор.

## Модель

Две ступени одной модели (`gdgen/model/`):
1. `<LAYOUT>` — весь геймплей уровня (скользящее окно с `<CONT>`).
2. `<DECO>` — декор и триггеры окнами по сегментам, с геймплеем вокруг (`<GAMEPLAY>`) в контексте.

`scripts/pretokenize.py` пишет потоки `layout`/`deco` в uint16-шарды, `scripts/train.py` учит (bf16/fp16, DDP),
`scripts/generate.py` генерирует уровень и проверяет раскладку симулятором.

**Музыка.** После каждого `SEG:s` идут токены песни на отрезке времени этого сегмента (`gdgen/music.py`):
громкость `MU:e`, плотность атак `MU:o`, бас `MU:l`, доли `MU:b` (клетка в 30 юнитов). Время считается через
порталы скорости до сегмента и сдвиг песни `kA13`. Признаки песен (librosa) считает `scripts/song_features.py`
на Kaggle, URL — `scripts/fetch_song_urls.py`; `pretokenize.py --features DIR` их вставляет. При генерации
`--song-features song.npz --song-offset 12` подставляет те же токены после каждого сегмента, который выбрала модель.

**Отбор симулятором.** `generate.py --select-segments 4` пишет раскладку кусками по 4 сегмента: после каждого
симулятор ищет путь до конца куска, непроходимый кусок перегенерируется (`--select-tries`), а если не проходит ни
одна попытка, переделывается и предыдущий (`--select-backtracks`). Итог — в `report.json` → `selection`.

## Оценка и галерея

```bash
python kaggle/kernel.py push eval --eval-gen gen,smoke     # метрики + симуляция против 300 featured (Kaggle CPU)
python kaggle/kernel.py output eval
python scripts/gallery.py --eval build/kernels/gd-levelgen-eval/out/eval --out gallery/smoke   # рендеры (локально)
```

`scripts/evaluate.py` считает для сгенерированных и случайных featured-уровней одни и те же метрики
(`gdgen/evaluate.py`): объекты по потокам, плотность по сегментам, пустые сегменты и самый длинный разрыв,
разброс по высоте, шипы/орбы/порталы/триггеры/группы/каналы, длительность, проходимость, а также JS-расстояние
между смесями id объектов. `summary.md` — медианы, 10–90 % и доля сгенерированных внутри диапазона featured.

## Симулятор

`gdgen/sim/` — физика GD 2.2081, сверенная с игрой по трассам (`re/physics_spec.md`). Python-версия — эталон;
`gdgen/sim/core/gdsim.cpp` — то же ядро на C++ (шаг, коллизии, триггеры, поиск пути), в 16–30 раз быстрее:

```python
from gdgen.sim.level import build_level
from gdgen.sim.fast import FastSimulator          # сам собирает ядро g++ при первом импорте
res = FastSimulator(build_level(header, objects)).find_path(max_frontier=256, time_limit=60)
```

`tests/test_fast_sim.py` требует побитового совпадения состояний C++ и Python на каждом шаге.

## Где считается

Локально ничего тяжёлого не запускается. Код и данные — приватные Kaggle-датасеты:

```bash
bash kaggle/push_code.sh                                   # kirillkholkin/gd-levelgen-code
bash kaggle/push_levels.sh data/snapshots/levels_X.sqlite  # снимок базы (VACUUM INTO) -> gd-levelgen-levels
python kaggle/kernel.py push tok --part 0/2                # тесты + токенизация на Kaggle (CPU)
python kaggle/kernel.py push smoke                         # короткое обучение + генерация (T4)
python kaggle/kernel.py status smoke ; python kaggle/kernel.py output smoke
```

Большое обучение — Colab-воркер (H100 / RTX 6000): открыть `colab/worker.ipynb`, задания ставятся локально:

```bash
python colab/jobs.py add --id m1 --gpu --timeout-h 20 --restore runs/m1 --save runs/m1 --save-every 30 \
    --results '$GD_RUNS/m1/log.jsonl' --cmd '$GD_PY scripts/train.py --data $GD_TOKENS --out $GD_RUNS/m1 --preset m'
python colab/jobs.py status ; python colab/jobs.py logs --id m1
```

## Тесты

```bash
python3 -m pytest -q tests      # на Kaggle: входят в kernel tok
```
