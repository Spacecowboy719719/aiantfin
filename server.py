import math
import os
import random
import threading
import time

from flask import Flask, jsonify, send_file

from brain import AntBrain
from config import HOME_X, HOME_Y, SERVER_HOST, SERVER_PORT, WORLD_HEIGHT, WORLD_WIDTH

currentdir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="static")

foods = []
foodclusters = []
hazards = []
poisonzones = []
walls = []

# Новая карта: регионы мира
region_labels = [
    {"x": 700,  "y": 1100, "text": "ЗАБЫТЫЙ АЛТАРЬ\nВЕЛИКАНОВ", "color": "#c0b090"},
    {"x": 1600, "y": 2400, "text": "ПОЛЕ ЗОЛОТОЙ\nРЫЦЫНА", "color": "#d4b878"},
    {"x": 2800, "y": 3100, "text": "БЕСКРАЙНИЕ ПОЛЯ\nТРАВЫ", "color": "#b8c9a8"},
    {"x": 2250, "y": 3200, "text": "МУРАВЬИНЫЕ\nТРОПЫ", "color": "#b0a080"},
    {"x": 3200, "y": 1500, "text": "СКОРЕЙСКИЙ\nПОДХОД", "color": "#a89880"},
    {"x": 3800, "y": 2200, "text": "КЛЮЧИВЫЙ\nАЗБУКА", "color": "#c8c0a0"},
    {"x": 2250, "y": 3500, "text": "СТОЛИЦА\nМУРАВЬЕВ", "color": "#e0d0a0"},
    {"x": 2000, "y": 4200, "text": "КРЫШИ\nУЩЕРБ ТИМ.", "color": "#b09080"},
    {"x": 500,  "y": 4500, "text": "ШЕНГУШИЙ ЛЕС\nГРИБОВ", "color": "#a0b878"},
    {"x": 1100, "y": 5000, "text": "НАД НЕБОЖНОЙ", "color": "#c8b898"},
    {"x": 2600, "y": 5500, "text": "КАМЕННЫЙ\nЛАБИРИНТ", "color": "#909090"},
    {"x": 3600, "y": 5000, "text": "ФОРТ ПЛАСТИКОВОЙ\nКРЫШКИ", "color": "#b09070"},
    {"x": 4200, "y": 1000, "text": "ЗАЛИВ\nСПИЧЕЧНОГО МОСТА", "color": "#90a0c0"},
    {"x": 3200, "y": 6500, "text": "БАТАРЕЙНАЯ\nПУСТИМЬ", "color": "#c0a880"},
    {"x": 1500, "y": 6200, "text": "ЧЕТВОРЬЕВСКИЙ\nГЕЛАДЬ", "color": "#b0a090"},
    {"x": 800,  "y": 6000, "text": "ПОТЕРЯВЫЕ\nПАЛЕВИНЫ МИКСИДОВ", "color": "#c0a080"},
    {"x": 4000, "y": 3500, "text": "ОБЛИВИАНА", "color": "#a0c0c0"},
    {"x": 2800, "y": 800,  "text": "ЯСНАЯ", "color": "#d0c8a0"},
    {"x": 1000, "y": 2800, "text": "ОПАСНОЕ\nКАССЕНОВОЕ", "color": "#c08080"},
    {"x": 3400, "y": 4200, "text": "ЦЕННЫЕ СПОРЫ", "color": "#b8c088"},
]

world_lock = threading.Lock()
ant = AntBrain()


def spawn_foods():
    global foods, foodclusters
    foods = []
    foodclusters = []

    for _ in range(12):
        while True:
            cx = random.uniform(250, WORLD_WIDTH - 250)
            cy = random.uniform(250, WORLD_HEIGHT - 250)
            if math.hypot(cx - HOME_X, cy - HOME_Y) > 420:
                break
        foodclusters.append({"x": cx, "y": cy})
        for _ in range(8):
            foods.append({
                "x": max(40, min(WORLD_WIDTH - 40, cx + random.uniform(-140, 140))),
                "y": max(40, min(WORLD_HEIGHT - 40, cy + random.uniform(-140, 140))),
            })


def spawn_hazards():
    global hazards
    hazards = [
        {"x": 1400, "y": 1800, "radius": 190},
        {"x": 3000, "y": 4200, "radius": 210},
        {"x": 2200, "y": 5800, "radius": 170},
    ]


def spawn_poison():
    global poisonzones
    poisonzones = [
        {"x": 980, "y": 3050, "radius": 150, "damage": 0.10},
        {"x": 3560, "y": 1680, "radius": 145, "damage": 0.09},
        {"x": 3080, "y": 6270, "radius": 165, "damage": 0.11},
    ]


def spawn_walls():
    global walls
    walls = [
        {"x": 820, "y": 1150, "w": 520, "h": 60},
        {"x": 2620, "y": 2100, "w": 80, "h": 700},
        {"x": 1450, "y": 4700, "w": 900, "h": 80},
        {"x": 3180, "y": 5200, "w": 460, "h": 60},
    ]


def push_out_of_walls():
    ax = ant.state["x"]
    ay = ant.state["y"]

    for wall in walls:
        inside = wall["x"] <= ax <= wall["x"] + wall["w"] and wall["y"] <= ay <= wall["y"] + wall["h"]
        if not inside:
            continue

        left = abs(ax - wall["x"])
        right = abs((wall["x"] + wall["w"]) - ax)
        top = abs(ay - wall["y"])
        bottom = abs((wall["y"] + wall["h"]) - ay)
        m = min(left, right, top, bottom)

        if m == left:
            ant.state["x"] = wall["x"] - 10
            ant.state["direction"] = 180
        elif m == right:
            ant.state["x"] = wall["x"] + wall["w"] + 10
            ant.state["direction"] = 0
        elif m == top:
            ant.state["y"] = wall["y"] - 10
            ant.state["direction"] = 270
        else:
            ant.state["y"] = wall["y"] + wall["h"] + 10
            ant.state["direction"] = 90

        ant.panictimer = max(ant.panictimer, 12)
        ant.state["fear"] = min(100.0, ant.state["fear"] + 10)


def apply_poison_damage():
    ax = ant.state["x"]
    ay = ant.state["y"]

    for p in poisonzones:
        d = math.hypot(ax - p["x"], ay - p["y"])
        if d < p["radius"]:
            ant.state["energy"] = max(0.0, ant.state["energy"] - p.get("damage", 0.08))
            ant.state["fear"] = min(100.0, ant.state["fear"] + 0.45)


def update_loop():
    while True:
        with world_lock:
            ant.perceive(foods, hazards, poisonzones, walls)
            ant.tick()
            push_out_of_walls()
            apply_poison_damage()

            ax = ant.state["x"]
            ay = ant.state["y"]

            eaten = []
            for i, food in enumerate(foods):
                if math.hypot(food["x"] - ax, food["y"] - ay) < 36:
                    eaten.append(i)

            if eaten:
                for i in reversed(eaten):
                    foods.pop(i)
                ant.state["energy"] = min(100.0, ant.state["energy"] + 8.5 * len(eaten))
                ant.state["hunger"] = max(0.0, ant.state["hunger"] - 16.0 * len(eaten))
                ant.iseatingnow = True
                ant.episodic.add("eat", ax, ay, {"count": len(eaten)})

            if len(foods) < 48 and random.random() < 0.06 and foodclusters:
                c = random.choice(foodclusters)
                foods.append({
                    "x": max(40, min(WORLD_WIDTH - 40, c["x"] + random.uniform(-160, 160))),
                    "y": max(40, min(WORLD_HEIGHT - 40, c["y"] + random.uniform(-160, 160))),
                })

        time.sleep(0.05)


@app.route("/")
def index():
    static_index = os.path.join(currentdir, "static", "index.html")
    if os.path.exists(static_index):
        return send_file(static_index)
    return send_file(os.path.join(currentdir, "index.html"))


@app.route("/state")
def get_state():
    with world_lock:
        st = ant.snapshot()
        st["foods"] = [dict(f) for f in foods]
        st["foodclusters"] = [dict(c) for c in foodclusters]
        st["hazards"] = [dict(h) for h in hazards]
        st["poisonzones"] = [dict(p) for p in poisonzones]
        st["walls"] = [dict(w) for w in walls]
        st["regions"] = region_labels
        return jsonify(st)


if __name__ == "__main__":
    spawn_foods()
    spawn_hazards()
    spawn_poison()
    spawn_walls()
    threading.Thread(target=update_loop, daemon=True).start()
    print(f"SERVER Starting AiAnt server on http://127.0.0.1:{SERVER_PORT}")
    print(f"SERVER World size {WORLD_WIDTH}x{WORLD_HEIGHT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)