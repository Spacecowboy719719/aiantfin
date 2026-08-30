import json
import math
import random
import threading
import time
from pathlib import Path

from config import (
    BRAIN_METAMONITOR_WINDOW,
    BRAIN_TOPOMAP_RESOLUTION,
    HOME_X,
    HOME_Y,
    SIMULATION_MAX_TOPO_NODES,
    SIMULATION_SAVE_INTERVAL_SEC,
    SIMULATION_STATE_VERSION,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from schemas import AntStateSchema, SnapshotResponse

STATE_FILE = Path(__file__).resolve().parent / "ant_state.json"


class TopoMap:
    def __init__(self, resolution=140, max_nodes=2400):
        self.res = resolution
        self.max_nodes = max_nodes
        self.nodes = {}

    def update(self, x, y, danger, fooddensity, iseating):
        gx, gy = int(x // self.res), int(y // self.res)
        node = self.nodes.setdefault(
            (gx, gy),
            {"visits": 0, "danger": 0.0, "food": 0.0, "ts": time.time()},
        )
        node["visits"] += 1
        node["ts"] = time.time()
        node["danger"] = max(node["danger"] * 0.97, min(1.0, float(danger)))

        if fooddensity > 0:
            node["food"] = min(1.0, node["food"] * 0.9 + 0.22 * float(fooddensity))
        elif not iseating and node["visits"] > 5:
            node["food"] = max(0.0, node["food"] - 0.03)

        if len(self.nodes) > self.max_nodes:
            drop = sorted(self.nodes.items(), key=lambda kv: (kv[1]["visits"], kv[1]["ts"]))[
                : max(12, self.max_nodes // 12)
            ]
            for key, _ in drop:
                self.nodes.pop(key, None)

    def get_best_food_node(self, ax, ay):
        best = None
        best_score = -10**9
        for (gx, gy), data in self.nodes.items():
            if data["food"] < 0.12:
                continue
            nx = gx * self.res + self.res * 0.5
            ny = gy * self.res + self.res * 0.5
            dist = math.hypot(nx - ax, ny - ay)
            if dist > 950:
                continue
            score = data["food"] * 1000 - dist - data["danger"] * 260
            if score > best_score:
                best_score = score
                best = (nx, ny)
        return best

    def get_unexplored_node(self, ax, ay):
        candidates = []
        for (gx, gy), data in self.nodes.items():
            if data["danger"] < 0.15 and data["visits"] < 3:
                nx = gx * self.res + self.res * 0.5
                ny = gy * self.res + self.res * 0.5
                d = math.hypot(nx - ax, ny - ay)
                candidates.append((d, nx, ny))
        if candidates:
            candidates.sort()
            return candidates[0][1], candidates[0][2]
        return None

    def export(self):
        out = []
        for (gx, gy), data in self.nodes.items():
            out.append([
                gx * self.res,
                gy * self.res,
                round(float(data["food"]), 3),
                round(float(data["danger"]), 3),
            ])
        return out

    def load(self, raw):
        self.nodes = {}
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            try:
                x = float(item[0])
                y = float(item[1])
                food = float(item[2])
                danger = float(item[3])
            except Exception:
                continue
            gx, gy = int(x // self.res), int(y // self.res)
            self.nodes[(gx, gy)] = {
                "visits": 1,
                "danger": max(0.0, min(1.0, danger)),
                "food": max(0.0, min(1.0, food)),
                "ts": time.time(),
            }


class MetaMonitor:
    def __init__(self, window=30):
        self.window = window
        self.history = []
        self.stuckcounter = 0

    def update(self, x, y, energy):
        self.history.append((x, y, energy))
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < self.window:
            return False
        ox, oy, _ = self.history[0]
        dx = abs(ox - x)
        dy = abs(oy - y)
        if dx < 10 and dy < 10:
            self.stuckcounter += 1
        else:
            self.stuckcounter = max(0, self.stuckcounter - 1)
        return self.stuckcounter > 5


class BodySchema:
    def __init__(self):
        self.lastangle = 0.0
        self.smoothness = 1.0

    def update(self, direction):
        diff = abs(direction - self.lastangle)
        if diff > 180:
            diff = 360 - diff
        self.smoothness = 0.9 * self.smoothness + 0.1 * (1.0 - diff / 180.0)
        self.lastangle = direction


class EpisodicMemory:
    def __init__(self, maxevents=32):
        self.maxevents = maxevents
        self.events = []

    def add(self, event_type, x, y, context=None):
        self.events.append({
            "type": str(event_type),
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "context": context or {},
            "time": round(time.time(), 3),
        })
        if len(self.events) > self.maxevents:
            self.events.pop(0)

    def export_labels(self, limit=6):
        return [e["type"] for e in self.events[-limit:]]

    def export(self, limit=24):
        return self.events[-limit:]

    def load(self, raw):
        self.events = []
        if not isinstance(raw, list):
            return
        for e in raw[-self.maxevents:]:
            if not isinstance(e, dict):
                continue
            self.events.append({
                "type": str(e.get("type", "event")),
                "x": round(float(e.get("x", 0.0)), 1),
                "y": round(float(e.get("y", 0.0)), 1),
                "context": e.get("context", {}) if isinstance(e.get("context", {}), dict) else {},
                "time": round(float(e.get("time", time.time())), 3),
            })


class PredictiveModel:
    def __init__(self):
        self.expectedenergy = 100.0
        self.error = 0.0

    def update(self, actualenergy):
        self.error = abs(self.expectedenergy - actualenergy)

    def expect(self, next_expected_energy):
        self.expectedenergy = next_expected_energy


class AntBrain:
    def __init__(self):
        self.lock = threading.RLock()

        self.worldw = WORLD_WIDTH
        self.worldh = WORLD_HEIGHT
        self.homex = HOME_X
        self.homey = HOME_Y

        self.speedpanic = 8.8
        self.speedescape = 7.1
        self.speedhunt = 5.8
        self.speedmemory = 4.6
        self.speedwander = 2.9
        self.speedsneak = 2.0
        self.speedrest = 1.0

        self.turnspeed = 7.0
        self.foodlockdistance = 500

        self.panictimer = 0
        self.iseatingnow = False
        self.last_hazard_turn = 0.0
        self.last_dangerlevel = 0.0
        self.last_wallhit = False
        self.last_save = time.time()
        self.last_mode = "wander"
        self.last_collision_log = 0.0

        self.last_target_sig = None
        self.last_target_dist = None
        self.no_progress_ticks = 0

        self.goal_hold_ticks = 0
        self.goal_hold_min = 25
        self.goal_hold_max = 50

        self.spatialmap = TopoMap(
            resolution=BRAIN_TOPOMAP_RESOLUTION,
            max_nodes=SIMULATION_MAX_TOPO_NODES,
        )
        self.meta = MetaMonitor(window=BRAIN_METAMONITOR_WINDOW)
        self.body = BodySchema()
        self.episodic = EpisodicMemory(maxevents=32)
        self.predictor = PredictiveModel()

        self.state = {
            "x": self.homex,
            "y": self.homey,
            "direction": random.uniform(0, 360),
            "speed": 0.0,
            "age": 0,
            "energy": 100.0,
            "hunger": 0.0,
            "fear": 0.0,
            "mode": "wander",
            "memory": [],
            "episodes": [],
            "spatialmap": [],
            "visiblefoodcount": 0,
            "target": None,
            "hazards": [],
            "curiosity": 0.0,
            "smoothness": 1.0,
            "dominant_drive": "curiosity",
            "drives": {},
            "decision_confidence": 0.0,
            "topo_nodes": 0,
            "state_version": SIMULATION_STATE_VERSION,
            "current_goal": "WANDER",
            "goal_utility": 0.0,
            "personality": {
                "fear_sensitivity": 1.0,
                "curiosity_boost": 0.02,
                "hunger_rate": 1.0,
                "exploration_boldness": 1.0,
            },
        }

        self.load_state()
        if not self.state.get("personality") or len(self.state["personality"]) < 4:
            self._randomize_personality()

    def _randomize_personality(self):
        self.state["personality"] = {
            "fear_sensitivity": round(random.uniform(0.7, 1.4), 2),
            "curiosity_boost": round(random.uniform(0.01, 0.05), 3),
            "hunger_rate": round(random.uniform(0.85, 1.2), 2),
            "exploration_boldness": round(random.uniform(0.5, 1.5), 2),
        }

    def dist(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def angleto(self, x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def normalizeangle(self, a):
        while a <= -180:
            a += 360
        while a > 180:
            a -= 360
        return a

    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def near_rect_point(self, x, y, rect):
        rx = rect["x"]
        ry = rect["y"]
        rw = rect["w"]
        rh = rect["h"]
        nx = min(max(x, rx), rx + rw)
        ny = min(max(y, ry), ry + rh)
        return nx, ny

    def vision_payload(self):
        mode = self.state["mode"]
        if mode in ("escape", "returnhome"):
            return {"range": 250, "angle": 155, "color": "#ff6f6f"}
        if mode in ("searchfood", "sneak"):
            return {"range": 270, "angle": 135 if mode == "searchfood" else 115, "color": "#58d5ff"}
        if mode == "rest":
            return {"range": 170, "angle": 180, "color": "#78c5ff"}
        if mode == "explore":
            return {"range": 290, "angle": 150, "color": "#d481ff"}
        return {"range": 250, "angle": 145, "color": "#7ef0a7"}

    def compute_drives(self):
        drives = {
            "hunger": round(float(self.state["hunger"]), 1),
            "fear": round(float(self.state["fear"]), 1),
            "curiosity": round(float(self.state.get("curiosity", 0.0)), 1),
            "rest": round(float(max(0.0, 100.0 - self.state["energy"])), 1),
        }
        dominant_drive = max(drives, key=drives.get) if drives else "curiosity"
        return drives, dominant_drive

    def load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            validated = AntStateSchema.model_validate(raw).model_dump()
        except Exception:
            return
        with self.lock:
            for key in self.state.keys():
                if key in validated:
                    self.state[key] = validated[key]
            self.spatialmap.load(raw.get("spatialmap", []))
            self.episodic.load(raw.get("episodes", []))

    def save_state(self):
        with self.lock:
            payload = {
                "state_version": SIMULATION_STATE_VERSION,
                "x": self.state["x"],
                "y": self.state["y"],
                "direction": self.state["direction"],
                "speed": self.state["speed"],
                "age": self.state["age"],
                "energy": self.state["energy"],
                "hunger": self.state["hunger"],
                "fear": self.state["fear"],
                "mode": self.state["mode"],
                "target": dict(self.state["target"]) if self.state["target"] else None,
                "spatialmap": self.spatialmap.export(),
                "episodes": self.episodic.export(limit=24),
                "dominant_drive": self.state.get("dominant_drive", "curiosity"),
                "drives": dict(self.state.get("drives", {})),
                "decision_confidence": self.state.get("decision_confidence", 0.0),
                "topo_nodes": self.state.get("topo_nodes", 0),
                "current_goal": self.state.get("current_goal", "WANDER"),
                "goal_utility": self.state.get("goal_utility", 0.0),
                "personality": dict(self.state["personality"]),
            }
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATE_FILE)

    def sense_food(self, foods):
        ax, ay = self.state["x"], self.state["y"]
        closest = None
        closestdist = 10**9
        visiblecount = 0
        self.iseatingnow = False
        for food in foods:
            d = self.dist(ax, ay, food["x"], food["y"])
            if d <= self.foodlockdistance:
                visiblecount += 1
            if d < closestdist:
                closestdist = d
                closest = food
            if d < 40:
                self.iseatingnow = True
        self.state["visiblefoodcount"] = visiblecount
        if closest is not None and self.state["hunger"] >= 18:
            self.state["target"] = {
                "kind": "food",
                "x": closest["x"],
                "y": closest["y"],
                "distance": closestdist,
            }
        elif closest is None and self.state.get("target") and self.state["target"].get("kind") == "food":
            self.state["target"] = None

    def sense_threats(self, hazards, poisonzones, walls):
        ax, ay = self.state["x"], self.state["y"]
        fearboost = 0.0
        turnpush = 0.0
        wallhit = False
        self.state["hazards"] = [dict(h) for h in hazards]
        merged = []
        for h in hazards:
            merged.append({"x": h["x"], "y": h["y"], "radius": h["radius"], "kind": "hazard"})
        for p in poisonzones:
            merged.append({"x": p["x"], "y": p["y"], "radius": p["radius"], "kind": "poison"})
        for h in merged:
            d = self.dist(ax, ay, h["x"], h["y"])
            safe = h["radius"] + (80 if h["kind"] == "hazard" else 95)
            if d < safe:
                fearboost = max(fearboost, (safe - d) * (0.42 if h["kind"] == "hazard" else 0.32))
                away = self.angleto(h["x"], h["y"], ax, ay)
                diff = self.normalizeangle(away - self.state["direction"])
                turnpush += max(-self.turnspeed, min(self.turnspeed, diff * 0.16))
            if d < h["radius"] + 18:
                self.panictimer = max(self.panictimer, 22 if h["kind"] == "poison" else 28)
                wallhit = True
        for wall in walls:
            nx, ny = self.near_rect_point(ax, ay, wall)
            d = self.dist(ax, ay, nx, ny)
            safe = 65
            inside = wall["x"] <= ax <= wall["x"] + wall["w"] and wall["y"] <= ay <= wall["y"] + wall["h"]
            if inside:
                d = 0.0
            if d < safe:
                away = self.angleto(nx, ny, ax, ay)
                diff = self.normalizeangle(away - self.state["direction"])
                gain = 0.24 if not inside else 0.52
                turnpush += max(-self.turnspeed * 1.6, min(self.turnspeed * 1.6, diff * gain))
                fearboost = max(fearboost, (safe - d) * (0.55 if inside else 0.24))
            if inside or d < 10:
                wallhit = True
                self.panictimer = max(self.panictimer, 18)
        fearboost *= self.state["personality"]["fear_sensitivity"]
        self.last_hazard_turn = self.clamp(turnpush, -self.turnspeed * 2.2, self.turnspeed * 2.2)
        self.last_dangerlevel = self.clamp(fearboost, 0.0, 100.0)
        self.last_wallhit = wallhit
        self.state["fear"] = min(100.0, max(self.state["fear"] * 0.96, fearboost))
        now = time.time()
        if wallhit and now - self.last_collision_log > 1.2:
            self.last_collision_log = now
            self.episodic.add("collision", ax, ay, {"danger": round(self.last_dangerlevel, 2)})

    def perceive(self, foods, hazards, poisonzones, walls):
        with self.lock:
            self.sense_food(foods)
            self.sense_threats(hazards, poisonzones, walls)

    def update_target_progress(self, ax, ay):
        target = self.state.get("target")
        if not target:
            self.last_target_sig = None
            self.last_target_dist = None
            self.no_progress_ticks = 0
            return
        sig = (
            target.get("kind", "none"),
            round(target.get("x", 0.0) / 30.0),
            round(target.get("y", 0.0) / 30.0),
        )
        dist = self.dist(ax, ay, target.get("x", ax), target.get("y", ay))
        if self.last_target_sig == sig:
            if self.last_target_dist is not None and dist >= self.last_target_dist - 0.8:
                self.no_progress_ticks += 1
            else:
                self.no_progress_ticks = max(0, self.no_progress_ticks - 2)
        else:
            self.no_progress_ticks = 0
        self.last_target_sig = sig
        self.last_target_dist = dist

    def moveforward(self):
        rad = math.radians(self.state["direction"])
        self.state["x"] += math.cos(rad) * self.state["speed"]
        self.state["y"] += math.sin(rad) * self.state["speed"]
        bounced = False
        if self.state["x"] < 8:
            self.state["x"] = 8
            self.state["direction"] = 0
            bounced = True
        elif self.state["x"] > self.worldw - 8:
            self.state["x"] = self.worldw - 8
            self.state["direction"] = 180
            bounced = True
        if self.state["y"] < 8:
            self.state["y"] = 8
            self.state["direction"] = 90
            bounced = True
        elif self.state["y"] > self.worldh - 8:
            self.state["y"] = self.worldh - 8
            self.state["direction"] = 270
            bounced = True
        if bounced:
            self.state["fear"] = min(100.0, self.state["fear"] + 15)
            self.panictimer = max(self.panictimer, 18)
            now = time.time()
            if now - self.last_collision_log > 1.2:
                self.last_collision_log = now
                self.episodic.add("hitwall", self.state["x"], self.state["y"], {"fear": round(self.state["fear"], 2)})

    def _compute_goal_utilities(self, ax, ay, disthome, inhome, visiblefood, best_mem_node, unexplored):
        hunger = self.state["hunger"]
        fear = self.state["fear"]
        curiosity = self.state["curiosity"]
        energy = self.state["energy"]
        person = self.state["personality"]

        rest_drive = max(0.0, 100.0 - energy) / 100.0

        goals = {
            "EAT_NEARBY": 0.0,
            "EAT_MEMORY": 0.0,
            "EXPLORE": 0.0,
            "RETURN_HOME": 0.0,
            "REST": 0.0,
            "WANDER": 0.05,
        }

        if visiblefood > 0 and self.state.get("target") and self.state["target"].get("kind") == "food":
            distfood = self.dist(ax, ay, self.state["target"]["x"], self.state["target"]["y"])
            goals["EAT_NEARBY"] = max(0.0, (hunger / 100.0) * 0.7 + (1.0 - min(distfood, self.foodlockdistance) / self.foodlockdistance) * 0.3)
        else:
            goals["EAT_NEARBY"] = 0.01

        if best_mem_node is not None:
            mem_quality = 0.6
            goals["EAT_MEMORY"] = (hunger / 100.0) * 0.8 + mem_quality * 0.2
        else:
            goals["EAT_MEMORY"] = 0.01

        explore_base = curiosity / 100.0
        if unexplored and curiosity > 10:
            explore_base += 0.25
        goals["EXPLORE"] = explore_base * 0.7 + (1.0 - fear / 100.0) * 0.15
        goals["EXPLORE"] *= person["exploration_boldness"]
        if hunger > 75 or fear > 60:
            goals["EXPLORE"] *= 0.15

        if fear > 15 or energy < 25:
            goals["RETURN_HOME"] = (fear / 100.0) * 0.8 + rest_drive * 0.2
        else:
            goals["RETURN_HOME"] = 0.02

        if inhome and energy < 90 and hunger < 60:
            goals["REST"] = rest_drive * 0.8 + (1.0 - hunger / 100.0) * 0.2
        else:
            goals["REST"] = 0.01

        temperature = 0.8
        utilities = list(goals.values())
        exp_utils = [math.exp(u / temperature) for u in utilities]
        sum_exp = sum(exp_utils)
        probabilities = [eu / sum_exp for eu in exp_utils]
        goal_names = list(goals.keys())
        chosen = random.choices(goal_names, weights=probabilities, k=1)[0]
        chosen_utility = goals[chosen]
        return chosen, chosen_utility

    def _should_recalc_goal(self, critical_condition):
        if critical_condition:
            return True
        self.goal_hold_ticks -= 1
        if self.goal_hold_ticks <= 0:
            self.goal_hold_ticks = random.randint(self.goal_hold_min, self.goal_hold_max)
            return True
        return False

    def tick(self):
        need_save = False
        with self.lock:
            self.state["age"] += 1
            ax, ay = self.state["x"], self.state["y"]

            if self.panictimer > 0:
                self.panictimer -= 1

            hazardturn = self.last_hazard_turn
            dangerlevel = self.last_dangerlevel
            wallhit = self.last_wallhit

            self.body.update(self.state["direction"])
            self.state["smoothness"] = round(self.body.smoothness, 3)

            if self.body.smoothness < 0.55 and self.panictimer <= 0:
                self.state["current_goal"] = "REST"
                self.goal_hold_ticks = 12
                self.state["fear"] = min(100.0, self.state["fear"] + 5)

            fooddensity = min(1.0, self.state["visiblefoodcount"] / 6.0)
            self.spatialmap.update(ax, ay, min(1.0, dangerlevel / 100.0), fooddensity, self.iseatingnow)
            self.state["spatialmap"] = self.spatialmap.export()

            isstuck = self.meta.update(ax, ay, self.state["energy"])
            self.predictor.update(self.state["energy"])

            self.state["curiosity"] = min(100.0,
                self.state["curiosity"] + self.state["personality"]["curiosity_boost"] +
                self.predictor.error * 5.0 + random.uniform(0.0, 3.0)
            )

            if self.predictor.error > 2.0:
                self.episodic.add("surprise", ax, ay, {"error": round(self.predictor.error, 2)})
                self.state["curiosity"] = min(100.0, self.state["curiosity"] + 12)
                self.state["personality"]["exploration_boldness"] = min(2.0, self.state["personality"]["exploration_boldness"] + 0.2)
            else:
                self.state["personality"]["exploration_boldness"] = max(0.5, self.state["personality"]["exploration_boldness"] - 0.01)

            disthome = self.dist(ax, ay, self.homex, self.homey)
            inhome = disthome < 150

            if inhome:
                self.state["fear"] = max(0.0, self.state["fear"] - 3.0)
                self.state["energy"] = min(100.0, self.state["energy"] + 0.36)

            critical = self.panictimer > 0 or wallhit or dangerlevel > 62
            if self._should_recalc_goal(critical):
                if critical:
                    self.state["current_goal"] = "ESCAPE_HOME"
                    self.state["goal_utility"] = 1.0
                else:
                    best_mem = self.spatialmap.get_best_food_node(ax, ay)
                    unexplored_node = self.spatialmap.get_unexplored_node(ax, ay) if random.random() < 0.7 else None
                    goal, util = self._compute_goal_utilities(
                        ax, ay, disthome, inhome,
                        self.state["visiblefoodcount"],
                        best_mem, unexplored_node
                    )
                    self.state["current_goal"] = goal
                    self.state["goal_utility"] = round(util, 4)

            current_goal = self.state["current_goal"]

            if current_goal in ("EXPLORE", "REST", "WANDER"):
                if self.state.get("target") and self.state["target"]["kind"] not in ("home", "memory"):
                    self.state["target"] = None
            elif current_goal == "ESCAPE_HOME":
                self.state["target"] = {"kind": "home", "x": self.homex, "y": self.homey, "distance": disthome}
            elif current_goal == "EAT_MEMORY":
                best_mem = self.spatialmap.get_best_food_node(ax, ay)
                if best_mem:
                    self.state["target"] = {
                        "kind": "memory",
                        "x": best_mem[0],
                        "y": best_mem[1],
                        "distance": self.dist(ax, ay, best_mem[0], best_mem[1]),
                    }
            elif current_goal == "EXPLORE" and unexplored_node:
                self.state["target"] = {
                    "kind": "explore",
                    "x": unexplored_node[0],
                    "y": unexplored_node[1],
                    "distance": self.dist(ax, ay, unexplored_node[0], unexplored_node[1]),
                }

            mode = "wander"
            desiredspeed = self.speedwander
            turn = hazardturn

            if self.panictimer > 0 or wallhit or dangerlevel > 62:
                mode = "escape"
                desiredspeed = self.speedpanic
                self.state["target"] = {"kind": "home", "x": self.homex, "y": self.homey, "distance": disthome}
            elif self.state["fear"] > 25 and not inhome:
                mode = "returnhome"
                desiredspeed = self.speedescape
                self.state["target"] = {"kind": "home", "x": self.homex, "y": self.homey, "distance": disthome}
            elif self.state["target"] and self.state["target"]["kind"] == "food":
                distfood = self.dist(ax, ay, self.state["target"]["x"], self.state["target"]["y"])
                self.state["target"]["distance"] = distfood
                if distfood < 150:
                    mode = "sneak"
                    desiredspeed = max(0.65, min(self.speedsneak, distfood * 0.018))
                else:
                    mode = "searchfood"
                    hungerboost = min(1.0, max(0.0, self.state["hunger"] - 25) / 75.0)
                    desiredspeed = self.speedwander + hungerboost * (self.speedhunt - self.speedwander)
            elif self.state["target"] and self.state["target"]["kind"] == "memory":
                mode = "searchfood"
                desiredspeed = self.speedmemory
            elif self.state["target"] and self.state["target"]["kind"] == "explore":
                mode = "explore"
                desiredspeed = self.speedwander * 1.1
            elif current_goal == "EXPLORE" and not inhome and self.state["energy"] > 35:
                mode = "explore"
                desiredspeed = self.speedwander * 1.08
            elif current_goal == "REST" and inhome and self.state["energy"] < 86:
                mode = "rest"
                desiredspeed = self.speedrest
                if self.state.get("target") and self.state["target"].get("kind") == "home":
                    self.state["target"] = None

            self.update_target_progress(ax, ay)

            if self.state["target"]:
                tx, ty = self.state["target"]["x"], self.state["target"]["y"]
                ang = self.angleto(ax, ay, tx, ty)
                diff = self.normalizeangle(ang - self.state["direction"])
                maxturn = self.turnspeed
                gain = 0.26
                if mode in ("escape", "returnhome"):
                    maxturn = self.turnspeed * 1.6
                    gain = 0.38
                elif self.state["target"]["kind"] == "food":
                    dist_to_target = self.dist(ax, ay, tx, ty)
                    if dist_to_target < 90:
                        maxturn = self.turnspeed * 3.3
                        gain = 0.80
                    elif dist_to_target < 160:
                        maxturn = self.turnspeed * 2.5
                        gain = 0.54
                    elif dist_to_target < 260:
                        maxturn = self.turnspeed * 1.6
                        gain = 0.35
                    if abs(diff) > 55 and dist_to_target < 130:
                        desiredspeed = min(desiredspeed, 1.15)
                    if abs(diff) > 35 and dist_to_target < 70:
                        desiredspeed = min(desiredspeed, 0.82)
                    if self.no_progress_ticks > 12:
                        desiredspeed = min(desiredspeed, 0.95)
                        turn += (1 if diff >= 0 else -1) * min(62, 8 + self.no_progress_ticks * 2.6)
                elif self.state["target"]["kind"] == "memory":
                    maxturn = self.turnspeed * 1.2
                    gain = 0.30
                elif self.state["target"]["kind"] == "explore":
                    maxturn = self.turnspeed * 1.0
                    gain = 0.22
                turn += max(-maxturn, min(maxturn, diff * gain))
            else:
                jitter = 3.0 if mode == "explore" else 2.0
                turn += random.uniform(-jitter, jitter)

            if isstuck:
                self.state["fear"] = min(100.0, self.state["fear"] + 22)
                self.meta.stuckcounter = 0
                self.episodic.add("stuck", ax, ay, {"energy": round(self.state["energy"], 2)})
                self.state["target"] = None
                turn += random.uniform(-95, 95)
                desiredspeed = min(desiredspeed, 1.0)

            self.state["mode"] = mode
            self.state["direction"] = (self.state["direction"] + turn) % 360

            accel = 0.18 if desiredspeed > self.state["speed"] else 0.34
            self.state["speed"] += (desiredspeed - self.state["speed"]) * accel

            self.moveforward()

            energydrain = 0.018 + 0.009 * (self.state["speed"] / max(self.speedpanic, 1.0))
            if mode == "rest":
                energydrain *= 0.45
            elif mode in ("escape", "returnhome"):
                energydrain *= 1.28
            elif mode == "explore":
                energydrain *= 1.08
            self.state["energy"] = max(0.0, self.state["energy"] - energydrain)
            self.state["hunger"] = min(100.0, self.state["hunger"] + (0.045 + self.state["speed"] * 0.004) * self.state["personality"]["hunger_rate"])

            if self.state["energy"] <= 0.0:
                self.episodic.add("reborn", self.state["x"], self.state["y"])
                self.state["x"] = self.homex
                self.state["y"] = self.homey
                self.state["direction"] = random.uniform(0, 360)
                self.state["speed"] = 0.0
                self.state["energy"] = 100.0
                self.state["hunger"] = 0.0
                self.state["fear"] = 0.0
                self.state["target"] = None
                self.panictimer = 0
                self._randomize_personality()
                self.state["current_goal"] = "WANDER"
                self.goal_hold_ticks = 0

            self.predictor.expect(self.state["energy"] - energydrain)
            self.state["memory"] = self.episodic.export_labels()
            self.state["episodes"] = self.episodic.export(limit=12)

            drives, _ = self.compute_drives()
            conf = max(
                0.0,
                min(
                    1.0,
                    0.34 + 0.26 * self.body.smoothness + 0.18 * (1.0 - min(1.0, self.state["fear"] / 100.0)) + 0.14 * (1.0 if self.state.get("target") else 0.0),
                ),
            )
            self.state["drives"] = drives
            goal_to_drive = {
                "EAT_NEARBY": "hunger",
                "EAT_MEMORY": "hunger",
                "EXPLORE": "curiosity",
                "RETURN_HOME": "fear",
                "ESCAPE_HOME": "fear",
                "REST": "rest",
                "WANDER": "curiosity",
            }
            self.state["dominant_drive"] = goal_to_drive.get(current_goal, "curiosity")
            self.state["decision_confidence"] = round(conf, 4)
            self.state["topo_nodes"] = len(self.spatialmap.nodes)

            if mode != self.last_mode:
                self.episodic.add("mode", self.state["x"], self.state["y"], {"mode": mode})
                self.last_mode = mode

            self.last_dangerlevel *= 0.94
            self.last_hazard_turn *= 0.7
            self.last_wallhit = False
            self.iseatingnow = False

            if time.time() - self.last_save > SIMULATION_SAVE_INTERVAL_SEC:
                self.last_save = time.time()
                need_save = True

        if need_save:
            self.save_state()

    def snapshot(self):
        with self.lock:
            raw = {
                "state_version": SIMULATION_STATE_VERSION,
                "x": self.state["x"],
                "y": self.state["y"],
                "direction": self.state["direction"],
                "speed": self.state["speed"],
                "age": self.state["age"],
                "energy": self.state["energy"],
                "hunger": self.state["hunger"],
                "fear": self.state["fear"],
                "mode": self.state["mode"],
                "memory": list(self.state["memory"]),
                "episodes": list(self.state["episodes"]),
                "spatialmap": list(self.state["spatialmap"]),
                "visiblefoodcount": self.state["visiblefoodcount"],
                "target": dict(self.state["target"]) if self.state["target"] else None,
                "hazards": [dict(h) for h in self.state["hazards"]],
                "curiosity": self.state["curiosity"],
                "smoothness": self.state["smoothness"],
                "dominant_drive": self.state["dominant_drive"],
                "drives": dict(self.state["drives"]),
                "decision_confidence": self.state["decision_confidence"],
                "decisionconfidence": self.state["decision_confidence"],
                "topo_nodes": self.state["topo_nodes"],
                "toponodes": self.state["topo_nodes"],
                "home": {"x": self.homex, "y": self.homey},
                "worldw": self.worldw,
                "worldh": self.worldh,
                "metastuckcounter": self.meta.stuckcounter,
                "meta_stuck_counter": self.meta.stuckcounter,
                "vision": self.vision_payload(),
                "current_goal": self.state["current_goal"],
                "goal_utility": self.state["goal_utility"],
                "personality": dict(self.state["personality"]),
            }
            return SnapshotResponse.model_validate(raw).model_dump()