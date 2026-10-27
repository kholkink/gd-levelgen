// C++ core of the GD 2.2081 simulator: one physics step and the path search, transcribed line by line from
// gdgen/sim/{physics,collide,triggers,search}.py (which document the derivation; spec sections are in
// re/physics_spec.md). The Python code stays the reference: tests/test_fast_sim.py steps both and requires
// bit-identical states, so keep the operations and their order exactly as in Python and build with
// -ffp-contract=off (no fused multiply-add).
//
// Level preparation (parsing, hitboxes, trigger lists) stays in Python (gdgen/sim/level.py); gdgen/sim/fast.py
// flattens it into the records below and talks to this file through the C functions at the end.
//
// Slopes follow gdgen/sim/collide.py (spec section 21 plus the trace-verified corrections noted there).
//
// Known, intended difference: when the search frontier overflows, Python orders states by repr(key) before
// thinning them and this file by the key's fields, so pruned searches may keep different (equally valid) states.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gd {

// ------------------------------------------------------------------------------------------------ constants
constexpr double STEP_SECONDS = 1.0 / 240.0;
constexpr double DT60 = 0.25;
constexpr double DTJ = 0.225;
constexpr double FLOOR_Y = 90.0;
constexpr double VEHICLE_GRAVITY = 0.958199;
constexpr double BROAD_PHASE_MARGIN = 45.0;
constexpr double BUCKET = 60.0;
constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;  // CPython math.radians: x * (pi / 180)
constexpr int TICKS_PER_SECOND = 240;
constexpr int MAX_EVENTS = 200000;

enum Mode { CUBE, SHIP, BALL, UFO, WAVE, ROBOT, SPIDER, SWING };
enum Cat { C_OTHER, C_SOLID, C_SLOPE, C_HAZARD, C_SPEED };
enum Type {
    T_OTHER, T_MODE_PORTAL, T_GRAV_INVERSE, T_GRAV_NORMAL, T_GRAV_TOGGLE, T_PAD_YELLOW, T_PAD_PINK, T_PAD_RED,
    T_GRAVITY_PAD, T_RING_YELLOW, T_RING_PINK, T_RING_RED, T_RING_GRAVITY, T_RING_GREEN, T_RING_DROP,
    T_RING_CUSTOM, T_RING_SPIDER, T_RING_DASH, T_RING_GRAVITY_DASH, T_RING_TELEPORT, T_TELEPORT_PORTAL,
    T_SPIDER_PAD, T_SIZE_REGULAR, T_SIZE_MINI, T_BREAKABLE, T_SLOPE, T_DUAL
};
enum TriggerKind { MOVE = 901, TOGGLE = 1049, SPAWN = 1268, STOP = 1616 };

inline bool is_fly(int m) { return m == SHIP || m == UFO || m == WAVE || m == SWING; }
inline bool is_bounded(int m) { return is_fly(m) || m == BALL || m == SPIDER; }  // == FIXED_GRAVITY_MODES
inline bool is_flyish5(int m) { return is_fly(m) || m == BALL; }
inline bool is_ring(int t) { return t >= T_RING_YELLOW && t <= T_RING_TELEPORT; }
inline bool is_pad(int t) { return t == T_PAD_YELLOW || t == T_PAD_PINK || t == T_PAD_RED; }
inline bool is_grav_portal(int t) { return t == T_GRAV_INVERSE || t == T_GRAV_NORMAL || t == T_GRAV_TOGGLE; }

inline double corridor_height(int m) {
    if (m == BALL) return 240.0;
    if (m == SPIDER) return 270.0;
    return 300.0;
}
inline double hitbox_size(int m) { return m == WAVE ? 10.0 : m == SPIDER ? 27.0 : 30.0; }
inline double bounds_height_of(int m) { return m == WAVE ? 20.0 : m == SPIDER ? 27.0 : 30.0; }
inline double ground_gravity_mult(int m) { return m == BALL || m == SPIDER ? 0.6 : m == ROBOT ? 0.9 : 1.0; }

struct SpeedC { double jump, g, xmul; };
inline SpeedC speed_constants(double speed) {
    if (speed == 0.7) return {10.620032, 0.940199, 5.980002};
    if (speed == 1.1) return {11.420032, 0.957199, 5.870002};
    if (speed == 1.3) return {11.230032, 0.961199, 6.000002};
    if (speed == 1.6) return {11.230032, 0.961199, 6.000002};
    return {11.180031776428223, 0.9581990242004395, 5.7700018882751465};  // 0.9
}
inline double x_step(double speed) { return speed * speed_constants(speed).xmul * 0.25; }

inline double py_trunc(double v) { return std::trunc(v); }      // int(v) for a float
inline double f32(double v) { return (double)(float)v; }        // level.f32: a C float
inline bool dir_uphill(int d) { return d == 0 || d == 3 || d == 6 || d == 7; }
inline bool dir_ceiling(int d) { return d == 1 || d == 3 || d == 5 || d == 6; }
inline double py_round(double v) { return std::nearbyint(v); }  // round(v): half to even

// -------------------------------------------------------------------------------------------- level records
#pragma pack(push, 8)
struct ObjRec {
    int32_t index, cat, type, gamemode;
    int32_t flip_x, flip_y, flipped, multi_activate, passable, free_mode, indexed, has_obb;
    int32_t t354, t345, t347, t351, t352, t353, t443, tg_ok;
    int32_t slope_dir, pad2;
    double x, y, x0, y0, x1, y1, radius, rotation, speed;
    double obb[8];
    double t51, t54, t346, t348, t349, t350, tg_x, tg_y, slope_theta;
};
struct SettingsRec {
    int32_t start_mode, start_mini, start_upside_down, fix_gravity_bug, fix_radius_collision, pad;
    double start_speed, spawn_y, max_gameplay_y, length;
};
struct TrigRec {
    int32_t kind, target, spawn_triggered, easing, lock_x, enable;
    double x, dx, dy, duration, rate, mod_x, delay;
};
struct SetRec {
    int32_t always, pad;
    double lo, hi;
};
#pragma pack(pop)

using Obj = ObjRec;

// ---------------------------------------------------------------------------------------------- speed logs
struct Break { long long tick; double x; double v; };

// Speed histories are interned: equal histories share an id, so states compare and hash them in O(1).
struct SpeedLogs {
    struct Node { int parent; Break b; };
    std::vector<Node> nodes;
    std::map<std::tuple<int, long long, uint64_t, uint64_t>, int> index;

    static uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
    int append(int parent, const Break& b) {
        auto key = std::make_tuple(parent, b.tick, bits(b.x), bits(b.v));
        auto it = index.find(key);
        if (it != index.end()) return it->second;
        nodes.push_back({parent, b});
        int id = (int)nodes.size() - 1;
        index.emplace(key, id);
        return id;
    }
    std::vector<Break> materialize(int id) const {
        std::vector<Break> out;
        for (; id >= 0; id = nodes[id].parent) out.push_back(nodes[id].b);
        std::reverse(out.begin(), out.end());
        return out;
    }
};

// ------------------------------------------------------------------------------------------------- triggers
struct Trigger {
    int kind, target;
    bool spawn_triggered, lock_x, enable;
    int easing;
    double x, dx, dy, duration, rate, mod_x, delay;
    std::vector<int> groups;
};

struct MoveSeg {
    long long start, ticks;
    double dx, dy;
    int easing;
    double rate;
    bool lock_x;
    double mod_x;
    long long stop = 1LL << 60;
};

struct GroupSet {
    std::vector<int> groups;
    std::vector<int> objects;  // indices into Level::objects
    double lo, hi;
    bool always;
};

double bounce(double u) {
    if (u < 1 / 2.75) return 7.5625 * u * u;
    if (u < 2 / 2.75) { u -= 1.5 / 2.75; return 7.5625 * u * u + 0.75; }
    if (u < 2.5 / 2.75) { u -= 2.25 / 2.75; return 7.5625 * u * u + 0.9375; }
    u -= 2.625 / 2.75;
    return 7.5625 * u * u + 0.984375;
}

double ease(int kind, double rate, double t) {
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    rate = rate > 0 ? rate : 2.0;
    if (kind == 1) {
        t *= 2;
        return t < 1 ? 0.5 * std::pow(t, rate) : 1.0 - 0.5 * std::pow(2.0 - t, rate);
    }
    if (kind == 2) return std::pow(t, rate);
    if (kind == 3) return std::pow(t, 1.0 / rate);
    if (kind == 4 || kind == 5 || kind == 6) {
        double period = rate;
        double s = period / 4.0;
        if (kind == 5) {
            t -= 1;
            return -(std::pow(2.0, 10 * t)) * std::sin((t - s) * 2 * PI / period);
        }
        if (kind == 6) return std::pow(2.0, -10 * t) * std::sin((t - s) * 2 * PI / period) + 1;
        t = t * 2 - 1;
        if (t < 0) return -0.5 * std::pow(2.0, 10 * t) * std::sin((t - s) * 2 * PI / period);
        return std::pow(2.0, -10 * t) * std::sin((t - s) * 2 * PI / period) * 0.5 + 1;
    }
    if (kind == 7 || kind == 8 || kind == 9) {
        if (kind == 8) return 1 - bounce(1 - t);
        if (kind == 9) return bounce(t);
        return t < 0.5 ? (1 - bounce(1 - 2 * t)) * 0.5 : bounce(2 * t - 1) * 0.5 + 0.5;
    }
    if (kind == 10) {
        t *= 2;
        return t < 1 ? 0.5 * std::pow(2.0, 10 * (t - 1)) : 0.5 * (2 - std::pow(2.0, -10 * (t - 1)));
    }
    if (kind == 11) return std::pow(2.0, 10 * (t - 1)) - 0.001;
    if (kind == 12) return 1 - std::pow(2.0, -10 * t);
    if (kind == 13) return -0.5 * (std::cos(PI * t) - 1);
    if (kind == 14) return 1 - std::cos(t * PI / 2);
    if (kind == 15) return std::sin(t * PI / 2);
    if (kind == 16 || kind == 17 || kind == 18) {
        double o = 1.70158;
        if (kind == 17) return t * t * ((o + 1) * t - o);
        if (kind == 18) { t -= 1; return t * t * ((o + 1) * t + o) + 1; }
        o *= 1.525;
        t *= 2;
        if (t < 1) return t * t * ((o + 1) * t - o) / 2;
        t -= 2;
        return t * t * ((o + 1) * t + o) / 2 + 1;
    }
    return t;
}

struct Timeline {
    std::vector<Break> breaks;
    std::vector<long long> break_ticks;
    std::unordered_map<int, std::vector<std::pair<long long, bool>>> enabled;
    std::unordered_map<int, std::vector<MoveSeg>> moves;
    std::unordered_map<long long, std::vector<Obj>> cache;  // tick -> dynamic objects (Timeline.cache)

    double x_at(long long tick) const {
        long long i = (long long)(std::upper_bound(break_ticks.begin(), break_ticks.end(), tick) - break_ticks.begin()) - 1;
        if (i < 0) i = 0;
        const Break& b = breaks[i];
        return b.x + (double)(tick - b.tick) * b.v;
    }
    long long tick_at(double x) const {
        for (size_t i = 0; i < breaks.size(); ++i) {
            bool last = i + 1 == breaks.size();
            if (last || x <= breaks[i + 1].x) {
                const Break& b = breaks[i];
                if (b.v > 0) return b.tick + std::max(0LL, (long long)std::ceil((x - b.x) / b.v));
                return b.tick;
            }
        }
        return breaks.back().tick;
    }
    bool group_enabled(int g, long long tick) const {
        auto it = enabled.find(g);
        if (it == enabled.end() || it->second.empty()) return true;
        const auto& ch = it->second;
        for (size_t i = ch.size(); i-- > 0;)
            if (ch[i].first <= tick) return ch[i].second;
        return true;
    }
    std::pair<double, double> group_offset(int g, long long tick) const {
        double ox = 0.0, oy = 0.0;
        auto it = moves.find(g);
        if (it == moves.end()) return {ox, oy};
        for (const MoveSeg& seg : it->second) {
            if (tick < seg.start) break;
            long long t = std::min(tick, seg.stop);
            double frac = seg.ticks <= 0 ? 1.0 : ease(seg.easing, seg.rate, (double)(t - seg.start) / (double)seg.ticks);
            ox += seg.dx * frac;
            oy += seg.dy * frac;
            if (seg.lock_x) {
                long long end = std::min(t, seg.start + seg.ticks);
                ox += (x_at(end) - x_at(seg.start)) * seg.mod_x;
            }
        }
        return {ox, oy};
    }
};

// -------------------------------------------------------------------------------------------------- level
struct Level {
    std::vector<Obj> objects;
    SettingsRec settings{};
    std::unordered_map<long long, std::vector<int>> bucket_index;
    std::unordered_map<uint64_t, std::vector<int>> query_cache;
    // trigger engine
    bool engine_sets = false;
    std::vector<Trigger> triggers;
    std::vector<int> x_triggers;
    std::unordered_map<int, std::vector<int>> by_group;
    std::vector<GroupSet> sets;
    std::vector<int> always_sets;
    std::unordered_map<long long, std::vector<int>> set_buckets;
    std::unordered_map<int, Timeline*> timelines;
    SpeedLogs logs;
    std::map<std::string, double> unsupported;

    ~Level() { for (auto& kv : timelines) delete kv.second; }

    void build_index() {
        bucket_index.clear();
        query_cache.clear();
        for (const Obj& o : objects) {
            if (!o.indexed) continue;
            for (long long b = (long long)std::floor(o.x0 / BUCKET); b <= (long long)std::floor(o.x1 / BUCKET); ++b)
                bucket_index[b].push_back(o.index);
        }
    }

    const std::vector<int>& query(double x0, double x1) {
        long long b0 = (long long)std::floor(x0 / BUCKET), b1 = (long long)std::floor(x1 / BUCKET);
        uint64_t key = ((uint64_t)(uint32_t)(int32_t)b0 << 32) | (uint32_t)(int32_t)b1;
        auto it = query_cache.find(key);
        if (it != query_cache.end()) return it->second;
        std::vector<int> out;
        for (long long b = b0; b <= b1; ++b) {
            auto f = bucket_index.find(b);
            if (f != bucket_index.end()) out.insert(out.end(), f->second.begin(), f->second.end());
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        if (query_cache.size() > 4096) query_cache.clear();
        return query_cache.emplace(key, std::move(out)).first->second;
    }

    void note(const std::string& reason, double x) {
        auto it = unsupported.find(reason);
        if (it == unsupported.end() || x < it->second) unsupported[reason] = x;
    }

    void finish_engine() {
        x_triggers.clear();
        by_group.clear();
        for (size_t i = 0; i < triggers.size(); ++i) {
            if (!triggers[i].spawn_triggered) x_triggers.push_back((int)i);
            for (int g : triggers[i].groups) by_group[g].push_back((int)i);
        }
        always_sets.clear();
        set_buckets.clear();
        for (size_t i = 0; i < sets.size(); ++i) {
            if (sets[i].always) { always_sets.push_back((int)i); continue; }
            for (long long b = (long long)std::floor(sets[i].lo / BUCKET); b <= (long long)std::floor(sets[i].hi / BUCKET); ++b)
                set_buckets[b].push_back((int)i);
        }
        engine_sets = !sets.empty();
    }

    std::vector<int> sets_near(double x0, double x1) const {
        std::vector<int> out(always_sets);
        std::vector<char> seen(sets.size(), 0);
        for (long long b = (long long)std::floor(x0 / BUCKET); b <= (long long)std::floor(x1 / BUCKET); ++b) {
            auto it = set_buckets.find(b);
            if (it == set_buckets.end()) continue;
            for (int s : it->second)
                if (!seen[s]) { seen[s] = 1; out.push_back(s); }
        }
        return out;
    }

    Timeline* timeline(int speed_log) {
        auto it = timelines.find(speed_log);
        if (it != timelines.end()) return it->second;
        Timeline* tl = new Timeline();
        tl->breaks = logs.materialize(speed_log);
        for (const Break& b : tl->breaks) tl->break_ticks.push_back(b.tick);
        run_timeline(*tl);
        timelines.emplace(speed_log, tl);
        return tl;
    }

    void run_timeline(Timeline& tl) {
        typedef std::tuple<long long, long long, int> Ev;  // (tick, seq, trigger)
        std::priority_queue<Ev, std::vector<Ev>, std::greater<Ev>> heap;
        long long seq = 0;
        for (int ti : x_triggers) heap.emplace(tl.tick_at(triggers[ti].x), seq++, ti);
        int processed = 0;
        while (!heap.empty() && processed < MAX_EVENTS) {
            Ev ev = heap.top();
            heap.pop();
            ++processed;
            long long tick = std::get<0>(ev);
            const Trigger& trig = triggers[std::get<2>(ev)];
            bool all_on = true;
            for (int g : trig.groups)
                if (!tl.group_enabled(g, tick)) { all_on = false; break; }
            if (!all_on) continue;
            if (trig.kind == TOGGLE) {
                tl.enabled[trig.target].push_back({tick, trig.enable});
            } else if (trig.kind == MOVE) {
                long long ticks = (long long)py_round(trig.duration * TICKS_PER_SECOND);
                tl.moves[trig.target].push_back({tick + 1, ticks, trig.dx, trig.dy, trig.easing, trig.rate, trig.lock_x,
                                                 trig.mod_x});
            } else if (trig.kind == STOP) {
                auto it = tl.moves.find(trig.target);
                if (it != tl.moves.end())
                    for (MoveSeg& seg : it->second)
                        if (seg.start <= tick && tick < seg.start + seg.ticks) seg.stop = tick;
            } else if (trig.kind == SPAWN) {
                long long at = tick + (long long)py_round(trig.delay * TICKS_PER_SECOND);
                auto it = by_group.find(trig.target);
                if (it != by_group.end())
                    for (int child : it->second) heap.emplace(at, seq++, child);
            }
        }
    }

    // Timeline.objects_near / _dynamic_at, including the per-tick cache filled by the first query of a tick.
    void dynamic_near(int speed_log, long long tick, double x0, double x1, std::vector<const Obj*>& out) {
        Timeline* tl = timeline(speed_log);
        auto it = tl->cache.find(tick);
        if (it == tl->cache.end()) {
            double wx0 = x0 - 120.0, wx1 = x1 + 120.0;
            std::vector<Obj> list;
            for (int si : sets_near(wx0, wx1)) {
                const GroupSet& gs = sets[si];
                bool all_on = true;
                for (int g : gs.groups)
                    if (!tl->group_enabled(g, tick)) { all_on = false; break; }
                if (!all_on) continue;
                double dx = 0.0, dy = 0.0;
                for (int g : gs.groups) {
                    auto off = tl->group_offset(g, tick);
                    dx += off.first;
                    dy += off.second;
                }
                for (int oi : gs.objects) {
                    const Obj& o = objects[oi];
                    if (o.x1 + dx < wx0 || o.x0 + dx > wx1) continue;
                    if (dx == 0.0 && dy == 0.0) { list.push_back(o); continue; }
                    Obj m = o;
                    m.x = o.x + dx; m.y = o.y + dy;
                    m.x0 = o.x0 + dx; m.y0 = o.y0 + dy; m.x1 = o.x1 + dx; m.y1 = o.y1 + dy;
                    if (o.has_obb)
                        for (int k = 0; k < 4; ++k) { m.obb[2 * k] = o.obb[2 * k] + dx; m.obb[2 * k + 1] = o.obb[2 * k + 1] + dy; }
                    list.push_back(m);
                }
            }
            if (tl->cache.size() > 64) tl->cache.clear();
            it = tl->cache.emplace(tick, std::move(list)).first;
        }
        for (const Obj& o : it->second)
            if (o.x0 <= x1 && o.x1 >= x0) out.push_back(&o);
    }
};

// ------------------------------------------------------------------------------------------------- player
struct Player {
    double x = 0.0, y = 105.0, prev_x = 0.0, prev_y = 105.0, y_vel = 0.0;
    int mode = CUBE;
    bool mini = false, upside_down = false;
    double speed = 0.9;
    bool held = false, buffer = false, ring_held_snap = false, ring_buf_snap = false, used_normal_ring = false;
    bool on_ground = true, grounded = true, is_rising = false, outside_clamp = false, jump_released = false;
    double robot_boost = 0.0, gravity_mod = 1.0;
    bool spider_pending = false;
    double spider_tp_time = -1.0;
    bool is_dashing = false;
    double dash_slope = 0.0;
    bool teleported = false, fix_gravity_bug = false;
    double time = 0.0;
    long long tick = 0;
    int speed_log = -1;
    double last_flip_time = 0.0, last_mode_change_time = 0.0;
    bool dead = false;
    int killer = -1;
    bool oob_prev = false;
    bool has_corridor = false;
    double cor_min = 0.0, cor_max = 0.0;
    bool free_mode = false;
    double corridor_prev_max = 0.0;
    bool has_pending_speed = false;
    double pending_speed = 0.0;
    std::vector<int> touched_rings;
    std::vector<int> activated;  // sorted, unique
    int prev_top_uid = -1, prev_bottom_uid = -1;
    // slopes (spec section 21)
    bool on_slope = false, was_on_slope = false;
    double slope_y_vel = 0.0, slope_angle = 0.0, slope_vs = 0.0;
    bool slope_ceil = false;
    int slope_uid = -1;
    bool slope_desc = false;
    double slope_enter_time = 0.0;
    bool slope_clamped = false;

    double s() const { return upside_down ? -1.0 : 1.0; }
    double vehicle_size() const { return mini ? 0.6 : 1.0; }
    double width() const { return hitbox_size(mode) * vehicle_size(); }
    double bounds_height() const { return bounds_height_of(mode); }
    SpeedC constants() const { return speed_constants(speed); }
    bool is_activated(int i) const { return std::binary_search(activated.begin(), activated.end(), i); }
    void activate(int i) {
        auto it = std::lower_bound(activated.begin(), activated.end(), i);
        if (it == activated.end() || *it != i) activated.insert(it, i);
    }
};

double set_y_vel(double v) {
    double t = py_trunc(v);
    if (v == t) return v;
    double frac = (v - t) * 1000.0;
    double rounded = py_trunc(frac + (frac >= 0 ? 0.5 : -0.5));
    return rounded / 1000.0 + t;
}

bool is_falling(const Player& p) {
    double g = p.constants().g;
    if (p.mode == SWING || p.fix_gravity_bug)
        return p.upside_down ? p.y_vel > -(2 * g) : (2 * g) > p.y_vel;
    if (p.upside_down) return p.y_vel > g + g;
    return (g + g) > p.y_vel;
}

void flip_gravity(Player& p, bool upside_down) {
    if (p.upside_down == upside_down) return;
    p.upside_down = upside_down;
    p.last_flip_time = p.time;
    p.on_ground = false;
    p.y_vel *= 0.5;
}

void update_jump(Player& p, double dt) {
    SpeedC c0 = p.constants();
    double jump_accel = c0.jump, g = c0.g;
    if (is_bounded(p.mode)) g = VEHICLE_GRAVITY;  // FIXED_GRAVITY_MODES
    double g0 = g;
    if (p.gravity_mod != 1.0) g *= p.gravity_mod;
    double sz = p.mini ? 0.8 : 1.0;
    double s = p.s();

    if (is_fly(p.mode)) {
        if (p.mini) sz = 0.85;
        double hi = 8.0 / sz, lo = -6.4 / sz;
        double v = p.y_vel;
        if ((s > 0 && ((0 <= v && v < hi) || (lo < v && v <= 0))) || (s < 0 && ((-hi < v && v <= 0) || (0 <= v && v < -lo))))
            p.outside_clamp = false;
        double c = 0.8;
        if (p.mode == SHIP) {
            double k = 0.8;
            if (p.held && !p.outside_clamp) k = -1.0;
            else if (p.outside_clamp && (s > 0 ? p.y_vel < 0 : p.y_vel > 0)) k = -1.0;
            if (!p.held && !is_falling(p)) k = 1.2;
            double gg = k >= 0 ? g : g0;
            double m = (p.held && is_falling(p)) ? 0.5 : 0.4;
            p.y_vel = set_y_vel(p.y_vel - s * dt * gg * k * m / sz);
        } else if (p.mode == UFO) {
            if (p.buffer && p.held) {
                p.buffer = false;
                double t = (p.mini ? 8.0 : 7.0) * sz;
                if (s > 0 ? p.y_vel < t : p.y_vel > -t) p.y_vel = set_y_vel(s * t);
            }
            double k = is_falling(p) ? 0.8 : 1.2;
            p.y_vel = set_y_vel(p.y_vel - s * dt * g * k * 0.5 / sz);
        } else if (p.mode == SWING) {
            if (p.buffer && p.held) {
                p.buffer = false;
                double v0 = p.y_vel;
                flip_gravity(p, !p.upside_down);
                p.y_vel = set_y_vel(v0 * 0.8);
            }
            sz = 1.0;
            c = 1.0;
            p.y_vel = set_y_vel(p.y_vel - p.s() * dt * g * (p.mini ? 0.6 : 0.4));
        } else {  // wave: yVel is cosmetic, motion comes from dx (spec section 3)
            p.y_vel = s * p.speed * p.constants().xmul * (p.held ? 1.0 : -1.0);
        }
        if (p.held && p.mode != WAVE) p.grounded = false;
        if (!p.outside_clamp && p.mode != WAVE) {
            if (p.s() > 0) p.y_vel = set_y_vel(std::min(std::max(p.y_vel, -8.0 * c / sz), 8.0 / sz));
            else p.y_vel = set_y_vel(std::min(std::max(p.y_vel, -8.0 / sz), 8.0 * c / sz));
        }
        if (is_falling(p)) p.is_rising = false;
        return;
    }

    double gm = ground_gravity_mult(p.mode);
    bool press = p.mode == ROBOT ? (p.buffer && p.held) : p.held;
    if (press && p.on_ground && !(p.is_dashing && p.mode != SPIDER)) {
        if (p.mode == SPIDER) {
            if (!p.is_dashing) p.spider_pending = true;
            return;
        }
        p.is_rising = true;
        p.grounded = p.on_ground = p.buffer = p.jump_released = false;
        p.robot_boost = 0.0;
        double v = jump_accel * (p.mode == ROBOT ? 0.5 : 1.0);
        p.y_vel = set_y_vel(s * v * sz);
        if ((p.was_on_slope || p.on_slope) && s * p.slope_y_vel > 0) {  // jump from a slope
            double cap = f32(p.y_vel * 1.4);
            double t = p.time - p.slope_enter_time;
            double f = t < 0.1 ? std::max(0.4, 10 * t) : 1.0;
            double b = p.mode == ROBOT ? p.slope_y_vel * f : p.slope_y_vel;
            p.y_vel = set_y_vel(p.y_vel + 0.25 * b);
            p.y_vel = set_y_vel(s > 0 ? std::min(p.y_vel, cap) : std::max(p.y_vel, cap));
        }
        if (p.mode == BALL) {
            flip_gravity(p, !p.upside_down);
            p.y_vel *= 0.6;
            p.held = false;
        }
        return;
    }
    if (p.is_rising) {
        if (p.mode == ROBOT && p.held && !p.jump_released && p.robot_boost < 1.5) {
            p.robot_boost += dt * 0.1;
            p.y_vel = set_y_vel(p.y_vel + s * dt * g * gm);
        }
        p.y_vel = set_y_vel(p.y_vel - s * dt * g * gm);
        if (is_falling(p)) p.is_rising = false;
        return;
    }
    if (is_falling(p)) p.on_ground = false;
    p.y_vel = set_y_vel(p.y_vel - s * dt * g * gm);
    p.y_vel = set_y_vel(s > 0 ? std::max(p.y_vel, -15.0) : std::min(p.y_vel, 15.0));
}

// ------------------------------------------------------------------------------------------------ geometry
struct Rect { double x0, y0, x1, y1; };

inline Rect player_rect(const Player& p, bool inner = false) {
    double w = inner ? p.width() / p.vehicle_size() * 0.3 : p.width();
    return {p.x - w / 2, p.y - w / 2, p.x + w / 2, p.y + w / 2};
}

inline bool overlaps(const Rect& r, const Obj& o) {
    return r.x0 <= o.x1 && o.x0 <= r.x1 && r.y0 <= o.y1 && o.y0 <= r.y1;
}

bool sat_overlap(const Rect& r, const double* c) {
    double rect_pts[8] = {r.x0, r.y0, r.x1, r.y0, r.x1, r.y1, r.x0, r.y1};
    double axes[12] = {1.0, 0.0, 0.0, 1.0};
    for (int i = 0; i < 4; ++i) {
        double ax = c[2 * i], ay = c[2 * i + 1], bx = c[2 * ((i + 1) % 4)], by = c[2 * ((i + 1) % 4) + 1];
        axes[4 + 2 * i] = -(by - ay);
        axes[5 + 2 * i] = bx - ax;
    }
    for (int a = 0; a < 6; ++a) {
        double nx = axes[2 * a], ny = axes[2 * a + 1];
        double amin = 0, amax = 0, bmin = 0, bmax = 0;
        for (int k = 0; k < 4; ++k) {
            double va = rect_pts[2 * k] * nx + rect_pts[2 * k + 1] * ny;
            double vb = c[2 * k] * nx + c[2 * k + 1] * ny;
            if (k == 0 || va < amin) amin = va;
            if (k == 0 || va > amax) amax = va;
            if (k == 0 || vb < bmin) bmin = vb;
            if (k == 0 || vb > bmax) bmax = vb;
        }
        if (amax < bmin || bmax < amin) return false;
    }
    return true;
}

bool object_hit(const Rect& r, const Obj& o, bool fix_radius = false) {
    if (o.radius != 0.0) {
        double cx = o.x, cy = o.y;
        if (fix_radius)
            return std::hypot((r.x0 + r.x1) / 2 - cx, (r.y0 + r.y1) / 2 - cy) <= (r.x1 - r.x0) / 2 + o.radius;
        if (r.x0 <= cx && cx <= r.x1 && r.y0 <= cy && cy <= r.y1) return true;
        double pts[8] = {r.x0, r.y0, r.x1, r.y0, r.x1, r.y1, r.x0, r.y1};
        for (int k = 0; k < 4; ++k)
            if (std::hypot(pts[2 * k] - cx, pts[2 * k + 1] - cy) < o.radius) return true;
        return false;
    }
    if (!overlaps(r, o)) return false;
    return o.has_obb ? sat_overlap(r, o.obb) : true;
}

double wrap180(double a) {
    a = std::fmod(a + 180.0, 360.0);
    if (a <= 0.0) a += 360.0;
    return a - 180.0;
}

void hit_ground(Player& p) {
    bool moving_away = p.s() * p.y_vel > 5;
    p.y_vel = 0.0;
    p.grounded = true;
    if (!moving_away) p.on_ground = true;
}

struct UidSet {  // a few object uids touched this step; fixed storage (no allocation in the hot loop)
    int v[32];
    int n = 0;
    bool contains(int uid) const {
        for (int i = 0; i < n; ++i)
            if (v[i] == uid) return true;
        return false;
    }
    void add(int uid) {
        if (!contains(uid) && n < 32) v[n++] = uid;
    }
    void clear() { n = 0; }
};

struct StepContacts {
    bool has_ceil = false, has_floor = false;
    double ceil = 0.0, floor = 0.0;
    int top_uid = -1, bottom_uid = -1;
    UidSet head_uids, feet_uids;

    void head(double y, int uid) {
        ceil = has_ceil ? std::min(ceil, y) : y;
        has_ceil = true;
        top_uid = uid;
        head_uids.add(uid);
    }
    void feet(double y, int uid) {
        floor = has_floor ? std::max(floor, y) : y;
        has_floor = true;
        bottom_uid = uid;
        feet_uids.add(uid);
    }
};

double pad_mod(int kind, int mode, bool mini) {
    if (kind == T_PAD_YELLOW) return 1.0;
    if (kind == T_PAD_PINK) {
        if (mode == SHIP) return 0.35;
        if (mode == UFO) return 0.40;
        if (mode == BALL || mode == SPIDER) return 0.70;
        return 0.65;
    }
    if (mode == SHIP) return mini ? 0.95 : 0.63;
    if (mode == UFO) return mini ? 0.98 : 0.60;
    return 1.25;
}

double ring_k(int kind, int mode, bool mini) {
    if (kind == T_RING_YELLOW) return mode == ROBOT ? 0.9 : 1.0;
    if (kind == T_RING_PINK) {
        if (mode == SHIP) return 0.37;
        if (mode == UFO) return 0.42;
        if (mode == BALL) return 0.77;
        return 0.72;
    }
    if (kind == T_RING_RED) {
        if (mode == SHIP) return mini ? 1.4 : 1.0;
        if (mode == UFO) return mini ? 1.36 : 1.02;
        if (mode == BALL) return 1.34;
        if (mode == ROBOT) return 1.28;
        if (mode == SPIDER) return 1.34;
        return 1.38;
    }
    if (kind == T_RING_GRAVITY) return 0.8;
    if (kind == T_RING_GREEN) return mode == SHIP ? 0.7 : 1.0;
    return 1.0;
}

// ---------------------------------------------------------------------------------------------- simulator
struct Simulator {
    Level& L;
    explicit Simulator(Level& level) : L(level) {}

    Player initial_state() {
        const SettingsRec& s = L.settings;
        Player p;
        p.mode = s.start_mode;
        p.mini = s.start_mini;
        p.speed = s.start_speed;
        p.upside_down = s.start_upside_down;
        p.y = p.prev_y = s.spawn_y;
        p.on_ground = p.grounded = false;
        p.fix_gravity_bug = s.fix_gravity_bug;
        p.speed_log = L.logs.append(-1, {0, p.x, x_step(p.speed)});
        if (is_bounded(p.mode)) {
            corridor(p.mode, p.y, p.cor_min, p.cor_max);
            p.has_corridor = true;
            p.corridor_prev_max = 0.0;
        }
        return p;
    }

    void near(const Player& p, double x0, double x1, std::vector<const Obj*>& out) {
        out.clear();
        for (int i : L.query(x0, x1)) out.push_back(&L.objects[i]);
        if (L.engine_sets) L.dynamic_near(p.speed_log, p.tick, x0, x1, out);
    }

    static double floor_y(const Player& p) { return FLOOR_Y + p.bounds_height() / 2 * p.vehicle_size(); }

    static void corridor(int mode, double portal_y, double& lo, double& hi) {
        double h = corridor_height(mode);
        double bottom = std::max(FLOOR_Y, std::floor((portal_y - h / 2) / 30.0) * 30.0);
        lo = bottom;
        hi = bottom + h;
    }

    Player step(const Player& state, bool hold) {
        Player p = state;
        p.time += STEP_SECONDS;
        if (p.has_pending_speed) {
            p.speed = p.pending_speed;
            p.has_pending_speed = false;
            p.pending_speed = 0.0;
            p.speed_log = L.logs.append(p.speed_log, {p.tick, p.x, x_step(p.speed)});
        }
        std::vector<int> press_used;
        if (hold && !p.held) {
            push_button(p, press_used);
        } else if (!hold && p.held) {
            p.held = p.buffer = false;
            p.jump_released = true;
            p.is_dashing = false;
        }
        move(p);
        p.tick += 1;
        check_collisions(p, press_used);
        if (!p.dead) post_slope(p);
        if (!p.activated.empty()) {
            double cutoff = p.x - 120.0;
            std::vector<int> keep;
            for (int i : p.activated)
                if (L.objects[i].x1 >= cutoff) keep.push_back(i);
            p.activated.swap(keep);
        }
        return p;
    }

    // physics.move, with the spider hook
    void move(Player& p) {
        p.y_vel = std::max(-1000.0, std::min(1000.0, p.y_vel));
        p.prev_x = p.x;
        p.prev_y = p.y;
        update_jump(p, DTJ);
        if (p.spider_pending) {
            p.spider_pending = false;
            spider_teleport(p, nullptr, false);
        }
        if (p.is_dashing) p.y_vel = 0.0;
        double dx = p.speed * p.constants().xmul * DT60;
        double dy;
        if (p.is_dashing) {
            dy = p.dash_slope * dx;
        } else if (p.mode == WAVE) {
            dy = std::fabs(dx) * p.s() * (p.held ? 1.0 : -1.0);
            if (p.mini) dy *= 2;
        } else {
            dy = DTJ * p.y_vel;
        }
        p.x += dx;
        p.y += dy;
        p.ring_held_snap = p.held;
        p.ring_buf_snap = p.buffer;
        p.used_normal_ring = false;
    }

    void push_button(Player& p, std::vector<int>& press_used) {
        p.held = p.buffer = true;
        p.ring_held_snap = p.ring_buf_snap = true;
        p.used_normal_ring = false;
        std::vector<int> rings;
        for (int i : p.touched_rings)
            if (!p.is_activated(i)) rings.push_back(i);
        if (!rings.empty()) {
            std::stable_sort(rings.begin(), rings.end(), [&](int a, int b) {
                return (L.objects[a].type != T_RING_CUSTOM) < (L.objects[b].type != T_RING_CUSTOM);
            });
            for (int i : rings) ring_jump(p, L.objects[i], press_used);
        } else if (p.mode == BALL || (!is_fly(p.mode) && p.held && p.on_ground)) {
            update_jump(p, 0.0);
        }
    }

    void check_collisions(Player& p, std::vector<int>& press_used) {
        bool oob_prev = p.oob_prev, oob = false;
        bool teleported = p.teleported;
        p.teleported = false;
        p.was_on_slope = p.on_slope;
        p.on_slope = false;
        StepContacts c;
        double fy = floor_y(p);
        bool bounded = is_bounded(p.mode) && !p.free_mode;
        if (p.y >= fy || bounded) {
            double v11 = (1.0 - p.vehicle_size()) * p.bounds_height() / 2;
            if (p.y > v11 + L.settings.max_gameplay_y) oob = true;
        } else {
            if (p.upside_down) {
                if (p.last_flip_time != 0.0 && p.time - p.last_flip_time < 0.1) {
                    p.y = fy;
                    hit_ground(p);
                    p.grounded = false;
                    p.touched_rings.clear();
                    finish(p, c, oob, oob_prev);
                    return;
                }
                oob = true;
            } else if (!p.is_rising) {
                p.y = fy;
                hit_ground(p);
                c.feet(FLOOR_Y, -2);
            }
        }
        if (bounded && p.has_corridor && !teleported) clamp_corridor(p, c);

        Rect r = player_rect(p);
        std::vector<const Obj*> solids, hazards, nearby;
        std::vector<int> touched;
        near(p, r.x0 - BROAD_PHASE_MARGIN, r.x1 + BROAD_PHASE_MARGIN, nearby);
        for (const Obj* op : nearby) {
            const Obj& o = *op;
            if (o.cat == C_SOLID) {
                if (!p.is_activated(o.index)) solids.push_back(op);
                continue;
            }
            if (o.cat == C_HAZARD) {
                hazards.push_back(op);
                continue;
            }
            if (p.is_activated(o.index) || !object_hit(r, o)) continue;
            if (o.cat == C_SLOPE) {
                slope(p, o, c);
                if (p.dead) return;
                r = player_rect(p);
                continue;
            }
            if (o.type == T_DUAL) { L.note("dual", o.x); continue; }
            activate(p, o, c, touched, press_used);
            r = player_rect(p);
        }
        p.touched_rings = touched;

        std::stable_sort(solids.begin(), solids.end(), [](const Obj* a, const Obj* b) { return -a->index < -b->index; });
        r = player_rect(p);
        for (const Obj* o : solids) {
            if (overlaps(r, *o)) {
                collide_solid(p, *o, c);
                if (p.dead) return;
                r = player_rect(p);
            }
        }
        r = player_rect(p);
        bool fix_radius = L.settings.fix_radius_collision;
        for (const Obj* o : hazards) {
            if (object_hit(r, *o, fix_radius)) {
                p.dead = true;
                p.killer = o->index;
                return;
            }
        }
        finish(p, c, oob, oob_prev);
    }

    void finish(Player& p, const StepContacts& c, bool oob, bool oob_prev) {
        if (c.has_ceil && c.has_floor && std::fabs(c.ceil - c.floor) < 0.7 * p.width()) { p.dead = true; p.killer = -2; }
        for (int i = 0; i < c.head_uids.n; ++i)
            if (c.feet_uids.contains(c.head_uids.v[i])) { p.dead = true; p.killer = -2; break; }
        if (oob && oob_prev) { p.dead = true; p.killer = -3; }
        p.oob_prev = oob;
        p.prev_top_uid = c.top_uid;
        p.prev_bottom_uid = c.bottom_uid;
    }

    void clamp_corridor(Player& p, StepContacts& c) {
        double min_y = p.cor_min, max_y = p.cor_max;
        double d_max = max_y - p.corridor_prev_max;
        p.corridor_prev_max = max_y;
        double half = p.bounds_height() / 2 * p.vehicle_size();
        double ceil_lim = max_y - half, floor_lim = min_y + half;
        bool ship_held = p.mode == SHIP && p.held;
        double old = p.y_vel;
        if (p.y <= ceil_lim) {
            if (p.y >= floor_lim) return;
            bool restore = (!p.upside_down) && ((p.mode != SHIP && p.mode != WAVE) || !p.buffer) && d_max != 0;
            p.was_on_slope = false;
            p.y = floor_lim;
            if ((p.mode != SHIP && p.mode != SWING) || p.y_vel < 0) hit_ground(p);
            c.feet(min_y, -2);
            if ((restore && p.y_vel > old) || (p.upside_down && ship_held && d_max < 0)) p.y_vel = old;
        } else {
            bool restore = p.upside_down && ((p.mode != SHIP && p.mode != WAVE) || !p.buffer) && d_max != 0;
            p.was_on_slope = false;
            p.y = ceil_lim;
            if (p.y_vel > 0) hit_ground(p);
            c.head(max_y, -3);
            if ((restore && old > p.y_vel) || (!p.upside_down && ship_held && d_max > 0)) p.y_vel = set_y_vel(old);
        }
    }

    void spider_teleport(Player& p, StepContacts* c, bool dynamic) {
        Rect r = player_rect(p);
        double h = r.y1 - r.y0;
        double px = p.x, py = p.y, s = p.s();
        double vs = p.vehicle_size();
        double bottom, top;
        if (is_bounded(p.mode) && p.has_corridor && !p.free_mode) {
            bottom = p.cor_min;
            top = p.cor_max;
        } else {
            bottom = std::max(FLOOR_Y, py - 3000.0);
            top = py + 3000.0;
        }
        double ay0, ay1;
        if (s > 0) {
            if (!dynamic) { ay0 = r.y1 - 2.0; ay1 = top; } else { ay0 = r.y1 + 2.0; ay1 = top + 4.0; }
        } else {
            if (!dynamic) { ay0 = bottom; ay1 = r.y0 + 2.0; } else { ay0 = bottom + 4.0; ay1 = r.y0 + 6.0; }
        }
        Rect area{px, ay0, r.x1 + 1.0, ay1};
        Rect strip{px - 4.0 * vs, ay0, px + 4.0 * vs, ay1};
        std::vector<const Obj*> nearby, solids, hazards;
        near(p, std::min(area.x0, strip.x0) - 60.0, std::max(area.x1, strip.x1) + 60.0, nearby);
        for (const Obj* o : nearby)
            if ((o->cat == C_SOLID || o->cat == C_SLOPE) && !p.is_activated(o->index) && overlaps(area, *o)) solids.push_back(o);
        for (const Obj* o : nearby)
            if (o->cat == C_HAZARD && object_hit(strip, *o)) hazards.push_back(o);
        if (solids.empty() && !hazards.empty() && !dynamic) {
            spider_teleport(p, c, true);
            return;
        }
        if (s > 0) {
            auto k = [](const Obj* o) { return (long long)(o->y0 * 10); };
            std::stable_sort(solids.begin(), solids.end(), [&](const Obj* a, const Obj* b) { return k(a) < k(b); });
            std::stable_sort(hazards.begin(), hazards.end(), [&](const Obj* a, const Obj* b) { return k(a) < k(b); });
        } else {
            auto k = [](const Obj* o) { return -(long long)(o->y1 * 10); };
            std::stable_sort(solids.begin(), solids.end(), [&](const Obj* a, const Obj* b) { return k(a) < k(b); });
            std::stable_sort(hazards.begin(), hazards.end(), [&](const Obj* a, const Obj* b) { return k(a) < k(b); });
        }
        double target = 0.0;
        for (size_t i = 0; i < solids.size(); ++i) {
            const Obj& o = *solids[i];
            double t = s > 0 ? o.y0 - h / 2 : o.y1 + h / 2;
            if ((s > 0 && t < py - 10) || (s < 0 && t > py + 10)) continue;
            if (o.x0 <= px) { target = t; break; }
            Rect moved{r.x0, t - h / 2, r.x1, t + h / 2};
            bool blocked = false;
            for (size_t j = 0; j < solids.size(); ++j) {
                const Obj& q = *solids[j];
                if (j != i && moved.x0 <= q.x1 && q.x0 <= moved.x1 && moved.y0 <= q.y1 - 2 && q.y0 + 2 <= moved.y1) {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) { target = t; break; }
        }
        if (solids.empty() && !hazards.empty()) {
            target = hazards[0]->y;
        } else {
            if (target == 0.0) target = s > 0 ? top - h / 2 : bottom + h / 2;
            if (!hazards.empty()) {
                double hy = hazards[0]->y;
                if ((s > 0 && hy < target) || (s < 0 && hy > target)) {
                    if (!dynamic) {
                        spider_teleport(p, c, true);
                        return;
                    }
                    target = hy;
                }
            }
        }
        p.y = target;
        flip_gravity(p, !p.upside_down);
        p.y_vel = set_y_vel(p.upside_down ? 1.0 : -1.0);
        p.held = false;
        p.spider_tp_time = p.time;
        p.prev_top_uid = p.prev_bottom_uid = -1;
        if (c) {
            c->head_uids.clear();
            c->feet_uids.clear();
        }
    }

    void teleport(Player& p, const Obj& o) {
        p.teleported = true;
        bool has_dest = false;
        double dx = 0, dy = 0;
        if (o.type == T_TELEPORT_PORTAL) {
            has_dest = true;
            dx = p.x;
            dy = o.y + o.t54;
        } else if (o.t51 > 0) {
            if (o.tg_ok) {
                has_dest = true;
                dx = o.tg_x;
                dy = o.tg_y;
                L.note("teleport to group (x jump)", o.x);
            }
        }
        if (has_dest) {
            if (o.t351) { dx -= o.x - p.x; dy -= o.y - p.y; }
            if (o.t352) dx = p.x;
            if (o.t353) dy = p.y;
            p.x = dx;
            p.y = dy;
            p.grounded = false;
        }
        int gravity = o.t354;
        if (gravity == 1) flip_gravity(p, false);
        else if (gravity == 2) flip_gravity(p, true);
        else if (gravity == 3) flip_gravity(p, !p.upside_down);
        double angle = (180.0 - o.rotation + (o.flip_x ? 180.0 : 0.0)) * DEG2RAD;
        if (o.t347) {
            double length = std::fabs(p.y_vel) * o.t350;
            if (o.t349 > 0) length = std::min(length, o.t349);
            if (o.t348 > 0) length = std::max(length, o.t348);
            p.y_vel = length * std::sin(angle);
            p.outside_clamp = true;
        } else if (o.t345) {
            double force = o.t346;
            bool additive = o.t443;
            if (force == 0.0 && !additive) {
                p.outside_clamp = false;
                p.y_vel = 0.0;
            } else {
                p.outside_clamp = true;
                p.y_vel = force * std::sin(angle) + (additive ? p.y_vel : 0.0);
            }
        }
    }

    void start_dash(Player& p, const Obj& o) {
        if (p.is_dashing) return;
        p.is_dashing = true;
        double a = o.rotation + (o.flip_x ? 180.0 : 0.0);
        double t = wrap180(std::fmod(-a, 360.0));
        if (std::fabs(t) > 90.0) t = wrap180((t > 0 ? 180.0 : -180.0) - t);
        t = std::max(-70.0, std::min(70.0, t));
        double rad = t * DEG2RAD;
        p.dash_slope = std::sin(rad) / std::fabs(std::cos(rad));
    }

    void consume(Player& p, const Obj& o) {
        if (!o.multi_activate) p.activate(o.index);
    }

    void activate(Player& p, const Obj& o, StepContacts& c, std::vector<int>& touched, std::vector<int>& press_used) {
        int t = o.type;
        if (is_grav_portal(t)) {
            bool target = t == T_GRAV_INVERSE ? true : t == T_GRAV_NORMAL ? false : !p.upside_down;
            consume(p, o);
            if (target != p.upside_down) {
                flip_gravity(p, target);
                c.has_ceil = c.has_floor = false;
                c.head_uids.clear();
                c.feet_uids.clear();
            }
        } else if (t == T_MODE_PORTAL) {
            consume(p, o);
            p.free_mode = o.free_mode;
            switch_mode(p, o.gamemode, o.y);
        } else if (is_pad(t)) {
            consume(p, o);
            propel(p, pad_mod(t, p.mode, p.mini));
            p.outside_clamp = t == T_PAD_RED;
            p.jump_released = true;
        } else if (t == T_GRAVITY_PAD) {
            bool pad_flipped = o.flipped;
            if (p.upside_down != (!pad_flipped)) {
                consume(p, o);
                propel(p, 0.8);
                flip_gravity(p, !pad_flipped);
                p.jump_released = true;
            }
        } else if (is_ring(t)) {
            touched.push_back(o.index);
            if (!is_fly(p.mode)) ring_jump(p, o, press_used);
        } else if (t == T_TELEPORT_PORTAL) {
            consume(p, o);
            teleport(p, o);
        } else if (t == T_SPIDER_PAD) {
            consume(p, o);
            p.jump_released = true;
            if ((bool)o.flipped != p.upside_down) flip_gravity(p, !p.upside_down);
            spider_teleport(p, &c, false);
        } else if (t == T_SIZE_REGULAR || t == T_SIZE_MINI) {
            consume(p, o);
            p.mini = t == T_SIZE_MINI;
        } else if (o.cat == C_SPEED) {
            consume(p, o);
            p.has_pending_speed = true;
            p.pending_speed = o.speed;
        }
    }

    void switch_mode(Player& p, int mode, double portal_y) {
        if (mode != p.mode) {
            if (is_fly(p.mode)) {
                p.y_vel *= 0.5;
                p.on_ground = p.grounded = false;
            }
            p.mode = mode;
            if (is_fly(mode)) {
                p.y_vel *= 0.5;
                p.on_ground = p.grounded = false;
            }
            if (mode == ROBOT || mode == SPIDER) p.robot_boost = 1.5;
            p.last_mode_change_time = p.time;
        }
        if (is_bounded(mode) && !p.free_mode) {
            corridor(mode, portal_y, p.cor_min, p.cor_max);
            p.has_corridor = true;
        } else {
            p.has_corridor = false;
            p.cor_min = p.cor_max = 0.0;
        }
    }

    void propel(Player& p, double mod) {
        p.is_rising = true;
        p.grounded = p.on_ground = false;
        p.y_vel = set_y_vel(p.s() * mod * 16.0 * (p.mini ? 0.8 : 1.0));
        if (p.mode == BALL || p.mode == SPIDER || p.mode == SWING) p.y_vel *= 0.6;
    }

    void ring_jump(Player& p, const Obj& o, std::vector<int>& press_used) {
        int t = o.type;
        if (p.dead || p.is_dashing || std::find(press_used.begin(), press_used.end(), o.index) != press_used.end() ||
            !p.ring_buf_snap || !p.ring_held_snap)
            return;
        if (t == T_RING_CUSTOM || p.used_normal_ring) return;
        if (!(t == T_RING_CUSTOM || t == T_RING_SPIDER || t == T_RING_DASH || t == T_RING_GRAVITY_DASH)) p.buffer = false;
        if (t == T_RING_SPIDER) {
            if ((bool)o.flipped != p.upside_down) flip_gravity(p, !p.upside_down);
            spider_teleport(p, nullptr, false);
        } else if (t == T_RING_DASH || t == T_RING_GRAVITY_DASH) {
            if (t == T_RING_GRAVITY_DASH) flip_gravity(p, !p.upside_down);
            start_dash(p, o);
        } else if (t == T_RING_TELEPORT) {
            teleport(p, o);
        } else if (t == T_RING_DROP) {
            double v = is_fly(p.mode) ? -14.0 : -15.0;
            if (p.mode == UFO) v *= 0.8;
            else if (p.mode == SPIDER) v *= 1.1;
            p.y_vel = set_y_vel(v * p.s());
            p.outside_clamp = true;
            if (p.mode == BALL || p.mode == SWING) p.held = false;
        } else {
            if (t == T_RING_GREEN) flip_gravity(p, !p.upside_down);
            double jump_accel = p.constants().jump;
            p.is_rising = true;
            p.grounded = p.on_ground = false;
            p.y_vel = set_y_vel(p.s() * jump_accel * ring_k(t, p.mode, p.mini) * (p.mini ? 0.8 : 1.0));
            if (t == T_RING_RED) p.outside_clamp = true;
            if (p.mode == BALL || p.mode == SPIDER) {
                p.y_vel *= 0.7;
                p.held = false;
            } else if (p.mode == SWING) {
                p.y_vel *= 0.6;
                p.held = false;
            }
            if (t == T_RING_GRAVITY) flip_gravity(p, !p.upside_down);
        }
        p.used_normal_ring = true;
        press_used.push_back(o.index);
        consume(p, o);
    }

    // ---------------------------------------------------------------------------------------- slopes (21)
    static double slope_y(const Obj& o, double x) {
        double k = (o.y1 - o.y0) / (o.x1 - o.x0);
        bool uphill = dir_uphill(o.slope_dir);
        if (x > o.x0) {
            double d = (x - o.x1) * k;
            return uphill ? o.y1 + d : o.y0 - d;
        }
        double d = (o.x0 - x) * k;
        return uphill ? o.y0 - d : o.y1 + d;
    }

    bool pre_slope(Player& p, const Obj& o, StepContacts& c) {
        if (o.index == p.slope_uid) return false;
        bool ceiling = dir_ceiling(o.slope_dir);
        Rect r = player_rect(p);
        int d = o.slope_dir;
        if ((d == 2 || d == 3 || d == 4 || d == 6) && p.x < o.x0) {
            Obj wall = o;
            wall.x1 = o.x0 + 1.0;
            wall.cat = C_SOLID; wall.type = T_OTHER; wall.passable = 0;
            if (overlaps(r, wall)) { collide_solid(p, wall, c); return true; }
        }
        if (ceiling ? p.y >= o.y1 : p.y <= o.y0 + 1) {
            double y0 = ceiling ? o.y1 - 1.0 : o.y0;
            Obj edge = o;
            edge.y0 = y0; edge.y1 = y0 + 1.0;
            edge.cat = C_SOLID; edge.type = T_OTHER; edge.passable = 0;
            if (overlaps(r, edge)) { collide_solid(p, edge, c); return true; }
        }
        return false;
    }

    void slope(Player& p, const Obj& o, StepContacts& c) {
        if (pre_slope(p, o, c) || p.dead) return;
        bool uphill = dir_uphill(o.slope_dir), ceil = dir_ceiling(o.slope_dir);
        bool descending = !uphill;
        if (p.upside_down) descending = !descending;
        double s = p.s();
        int mode = p.mode;
        double theta = o.slope_theta;
        double ow = o.x1 - o.x0, oh = o.y1 - o.y0;
        double spd = p.speed * p.constants().xmul;
        double vs = f32(oh / (ow / spd));
        if (p.was_on_slope && s * p.slope_y_vel > 0) {
            if (descending && p.slope_ceil == ceil && ceil == p.upside_down) return;
            if (p.slope_vs > vs) return;
            if (p.slope_clamped && o.index == p.slope_uid && !descending) return;
        }
        Rect r = player_rect(p);
        double h = r.y1 - r.y0;
        double px = p.x, py = p.y;
        double tol = descending ? (p.was_on_slope ? 4.0 : 1.0) : 0.0;
        if (p.was_on_slope) {
            double cos_a = std::cos(p.slope_angle);
            double probe = p.upside_down ? py + (h / 2) / cos_a + tol : py - (h / 2) / cos_a - tol;
            if ((s > 0 && probe > o.y1) || (s < 0 && probe < o.y0)) return;
        } else if (!(r.x0 <= o.x1 && o.x0 <= r.x1 && r.y0 <= o.y1 - 1 && o.y0 + 1 <= r.y1)) {
            return;
        }
        double hc = (h / std::cos(theta)) / 2;
        bool trans = p.was_on_slope && p.slope_uid != o.index && p.slope_ceil != ceil;
        double lift = trans ? 20 * p.vehicle_size() : 0.0;
        double ys = slope_y(o, px);
        double target, raw;
        bool clamped;
        if (!ceil) {
            raw = ys + hc - lift;
            target = std::min(std::max(raw, o.y0), o.y1 + h / 2);
            clamped = raw > o.y1 + h / 2;
        } else {
            raw = ys - hc + lift;
            target = std::min(std::max(raw, o.y0 - h / 2), o.y1);
            clamped = raw < o.y0 - h / 2;
        }
        bool feet_side = ceil == p.upside_down;
        bool fly4 = is_fly(mode);
        bool fly_held = false, hit;
        if (feet_side) {
            bool snap_tol = descending && !trans && !p.on_slope &&
                            (mode == SHIP ? !p.held : (!p.is_rising || mode == UFO || mode == WAVE || mode == SWING));
            if (s > 0) hit = py < target || (snap_tol && py < target + tol && !(mode == UFO && p.y_vel > 0));
            else hit = py > target || (snap_tol && py > target - tol && !(mode == UFO && p.y_vel < 0));
            fly_held = p.held && fly4 && !descending;
        } else {
            double tol2 = descending ? 0.0 : ((fly4 && p.held) ? 1.0 + (p.was_on_slope ? 1.0 : 0.0) : 0.0);
            bool ok = !trans && (!fly4 || !descending) && !p.on_slope;
            hit = s > 0 ? (py > target || (ok && py > target - tol2)) : (py < target || (ok && py < target + tol2));
            if (hit && (mode == CUBE || mode == ROBOT || mode == SPIDER)) {
                if (p.time - p.last_mode_change_time < 0.1 || p.time - p.last_flip_time < 0.1) {
                    p.y = target;
                    p.y_vel = s > 0 ? std::min(p.y_vel, -2.0) : std::max(p.y_vel, 2.0);
                    p.on_ground = p.grounded = false;
                    return;
                }
                if (s > 0 ? py - 2 <= target : py + 2 >= target) {
                    p.on_ground = p.grounded = false;
                    return;
                }
                p.dead = true;
                p.killer = o.index;
                return;
            }
        }
        if (!hit) return;
        if (trans || mode == WAVE) { p.dead = true; p.killer = o.index; return; }
        if (descending && mode == WAVE && p.held && ceil == p.upside_down) return;
        p.slope_uid = o.index;
        p.slope_ceil = ceil;
        p.slope_desc = descending;
        p.slope_clamped = clamped;
        p.slope_angle = (s * theta) * (descending ? 1.0 : -1.0);
        p.y = target;
        if (!p.was_on_slope) p.slope_enter_time = p.time;
        p.on_slope = true;
        if (feet_side && !fly_held) {
            double v = p.y_vel;
            hit_ground(p);
            if (s * v > 5) p.y_vel = set_y_vel(v);
            if (s > 0) c.feet(target - s * h / 2, o.index); else c.head(target - s * h / 2, o.index);
        }
        if (!feet_side && !(!p.held && is_flyish5(mode) && descending && p.was_on_slope)) {
            p.y_vel = s > 0 ? std::min(p.y_vel, 0.0) : std::max(p.y_vel, 0.0);
            p.on_ground = p.grounded = false;
            if (s > 0) c.head(target + s * h / 2, o.index); else c.feet(target + s * h / 2, o.index);
        }
        if (!feet_side && fly4 && descending && !p.held) p.y_vel = s > 0 ? std::min(p.y_vel, -2.0) : std::max(p.y_vel, 2.0);
        if (fly_held) p.y_vel = s > 0 ? std::max(p.y_vel, 2.0) : std::min(p.y_vel, -2.0);
        p.slope_vs = vs;
        p.slope_y_vel = f32(s * (descending ? -1.0 : 1.0) * std::min(1.1, 0.8 / theta) * vs * 1.4);
        if (is_flyish5(mode)) p.slope_y_vel = f32(p.slope_y_vel * 0.75);
    }

    void post_slope(Player& p) {
        if (p.on_slope || !p.was_on_slope) return;
        double s = p.s();
        bool c = p.slope_ceil;
        double v_slope = p.slope_y_vel;
        bool launch = (v_slope > 0 && v_slope > p.y_vel && !c) || (v_slope < 0 && p.y_vel > v_slope && c);
        if (launch && (!p.is_rising || is_flyish5(p.mode))) {
            double t = p.time - p.slope_enter_time;
            double v = v_slope * (t < 0.1 ? std::max(0.4, 10 * t) : 1.0);
            bool apply;
            if (!p.upside_down) apply = (c && p.slope_desc) ? v < p.y_vel : v > p.y_vel;
            else apply = (!c && p.slope_desc) ? v > p.y_vel : v < p.y_vel;
            if (apply) {
                p.is_rising = true;
                p.grounded = p.on_ground = false;
                p.outside_clamp = true;
                p.y_vel = set_y_vel(v);
            }
        } else if ((s > 0 && v_slope < 0) || (s < 0 && v_slope > 0)) {
            if (!p.is_rising && !p.held && !p.buffer && c == p.upside_down) {
                p.grounded = p.on_ground = false;
                p.y_vel = set_y_vel(-s * p.slope_vs);
            }
        }
        p.slope_vs = p.slope_y_vel = p.slope_angle = 0.0;
        p.slope_uid = -1;
        p.slope_clamped = false;
    }

    void collide_solid(Player& p, const Obj& o, StepContacts& c) {
        double s = p.s();
        double h = p.width();
        double dy = p.y - p.prev_y;
        bool fly = is_fly(p.mode);
        double tol = fly ? 6.0 : 10.0;
        double feet = p.y - s * h / 2 + s * tol;
        double feet_prev = feet - dy;
        double head = p.y + s * h / 2 - s * tol;
        double head_prev = head - dy;
        bool flyish = fly || p.mode == BALL;
        bool passable = o.passable;
        bool flag71 = s > 0 ? (o.y0 >= feet || o.y0 >= feet_prev) : (feet >= o.y1 || feet_prev >= o.y1);

        if (p.mode != WAVE) {
            bool below = s > 0 ? (feet >= o.y1 || feet_prev >= o.y1) : (head >= o.y1 || head_prev >= o.y1);
            if (((!p.upside_down) || (flyish && !passable)) && below) {
                if (p.y_vel <= 0) {
                    p.y = o.y1 + h / 2;
                    if (flag71) p.y_vel = 0.0;
                    else hit_ground(p);
                    if (s > 0) c.feet(o.y1, o.index); else c.head(o.y1, o.index);
                }
                return;
            }
            if (p.upside_down || (flyish && !passable)) {
                double cmp = s > 0 ? head : feet, cmp_prev = s > 0 ? head_prev : feet_prev;
                bool deep = o.y0 < cmp && o.y0 < cmp_prev;
                int prev_uid = s > 0 ? p.prev_top_uid : p.prev_bottom_uid;
                if (!deep || o.index == prev_uid) {
                    if (p.y_vel >= 0) {
                        p.y = o.y0 - h / 2;
                        if (flag71) p.y_vel = 0.0;
                        else hit_ground(p);
                        if (!((passable || o.type == T_BREAKABLE) && flag71 && s > 0)) {
                            if (s > 0) c.head(o.y0, o.index); else c.feet(o.y0, o.index);
                        }
                    }
                    return;
                }
            }
        }
        if (passable) return;
        Rect inner = player_rect(p, true);
        if (!overlaps(inner, o)) return;
        if (p.mode == SPIDER && inner.x1 > o.x1 && p.time - p.spider_tp_time < 0.04) return;
        if (flag71 && p.last_flip_time != 0.0 && p.time - p.last_flip_time < 0.1) {
            p.y = p.upside_down ? o.y1 + h / 2 : o.y0 - h / 2;
            hit_ground(p);
            p.grounded = false;
            return;
        }
        if (o.type == T_BREAKABLE) {
            p.activate(o.index);
            return;
        }
        p.dead = true;
        p.killer = o.index;
    }
};

// ------------------------------------------------------------------------------------------------- search
struct Key {
    long long yq, vq, boost_q, slope_q, syv_q;
    int mode, speed_log, slope_uid;
    uint64_t speed_bits, pending_bits, cmin_bits, cmax_bits;
    uint32_t flags;
    std::vector<int> touched, activated;

    bool operator==(const Key& o) const {
        return yq == o.yq && vq == o.vq && boost_q == o.boost_q && slope_q == o.slope_q && mode == o.mode &&
               syv_q == o.syv_q && slope_uid == o.slope_uid &&
               speed_log == o.speed_log && speed_bits == o.speed_bits && pending_bits == o.pending_bits &&
               cmin_bits == o.cmin_bits && cmax_bits == o.cmax_bits && flags == o.flags && touched == o.touched &&
               activated == o.activated;
    }
    bool operator<(const Key& o) const {
        return std::tie(yq, vq, mode, flags, speed_bits, speed_log, boost_q, slope_q, syv_q, slope_uid, pending_bits,
                        cmin_bits, cmax_bits, touched, activated) <
               std::tie(o.yq, o.vq, o.mode, o.flags, o.speed_bits, o.speed_log, o.boost_q, o.slope_q, o.syv_q,
                        o.slope_uid, o.pending_bits, o.cmin_bits, o.cmax_bits, o.touched, o.activated);
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        mix((uint64_t)k.yq); mix((uint64_t)k.vq); mix((uint64_t)k.boost_q); mix((uint64_t)k.slope_q);
        mix((uint64_t)k.syv_q); mix((uint64_t)(k.slope_uid + 2));
        mix((uint64_t)k.mode); mix((uint64_t)k.speed_log); mix(k.speed_bits); mix(k.pending_bits);
        mix(k.cmin_bits); mix(k.cmax_bits); mix(k.flags);
        for (int t : k.touched) mix((uint64_t)t + 0x1000);
        for (int a : k.activated) mix((uint64_t)a + 0x2000);
        return (size_t)h;
    }
};

inline uint64_t dbits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }

// Simulator.key: coarse on purpose; merged states keep one exact representative, so a found path is still exact.
Key make_key(const Player& p) {
    Key k;
    k.yq = (long long)py_round(p.y * 2.0);
    k.vq = (long long)py_round(p.y_vel * 4.0);
    k.boost_q = (long long)py_round(p.robot_boost * 100.0);
    k.slope_q = (long long)py_round(p.dash_slope * 1000.0);
    k.syv_q = (long long)py_round(p.slope_y_vel * 100.0);
    k.slope_uid = p.slope_uid;
    k.mode = p.mode;
    k.speed_log = p.speed_log;
    k.speed_bits = dbits(p.speed);
    k.pending_bits = p.has_pending_speed ? dbits(p.pending_speed) : 0x7ff8dead00000000ULL;
    k.cmin_bits = p.has_corridor ? dbits(p.cor_min) : 0x7ff8dead00000001ULL;
    k.cmax_bits = p.has_corridor ? dbits(p.cor_max) : 0x7ff8dead00000001ULL;
    uint32_t f = 0;
    int bit = 0;
    for (bool b : {p.mini, p.upside_down, p.held, p.buffer, p.on_ground, p.is_rising, p.outside_clamp, p.oob_prev,
                   (bool)(p.time - p.last_flip_time < 0.1), p.is_dashing, p.teleported, p.on_slope, p.slope_ceil,
                   p.slope_clamped})
        f |= (uint32_t)b << bit++;
    k.flags = f;
    k.touched = p.touched_rings;
    k.activated = p.activated;
    return k;
}

struct BucketKey {
    int mode;
    bool upside, mini;
    uint64_t speed_bits;
    long long band;
    bool operator==(const BucketKey& o) const {
        return mode == o.mode && upside == o.upside && mini == o.mini && speed_bits == o.speed_bits && band == o.band;
    }
};
struct BucketHash {
    size_t operator()(const BucketKey& b) const {
        return std::hash<long long>()(b.band * 1315423911LL + b.mode * 7 + b.upside * 3 + b.mini) ^ (size_t)b.speed_bits;
    }
};

// Simulator.bucket
BucketKey make_bucket(const Player& p) {
    return {p.mode, p.upside_down, p.mini, dbits(p.speed), (long long)std::floor(p.y / 10.0)};
}

struct SearchResult {
    bool passed = false;
    double max_x = 0.0;
    int ticks = 0;
    std::vector<uint8_t> inputs;
    int frontier_peak = 0;
};

struct Node { Player st; int node; Key key; };

// n positions evenly spaced over [0, size)
static std::vector<size_t> spread(const std::vector<size_t>& items, size_t n) {
    std::vector<size_t> out;
    out.reserve(n);
    double stride = (double)items.size() / (double)n;
    for (size_t i = 0; i < n; ++i) out.push_back(items[(size_t)(i * stride)]);
    return out;
}

// search._prune, on indices (nodes are large; only the kept ones are moved)
static std::vector<Node> prune(std::vector<Node>& frontier, size_t max_frontier, bool use_bucket) {
    std::vector<size_t> order(frontier.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return frontier[a].key < frontier[b].key; });
    std::vector<size_t> pick;
    if (!use_bucket) {
        pick = spread(order, max_frontier);
    } else {
        std::vector<BucketKey> groups_order;
        std::unordered_map<BucketKey, std::vector<size_t>, BucketHash> groups;
        for (size_t i : order) {
            BucketKey b = make_bucket(frontier[i].st);
            auto it = groups.find(b);
            if (it == groups.end()) { groups_order.push_back(b); groups[b].push_back(i); }
            else it->second.push_back(i);
        }
        size_t quota = std::max<size_t>(1, max_frontier / groups.size());
        std::vector<size_t> kept, rest;
        for (const BucketKey& b : groups_order) {
            const std::vector<size_t>& g = groups[b];
            if (g.size() <= quota) {
                kept.insert(kept.end(), g.begin(), g.end());
            } else {
                double stride = (double)g.size() / (double)quota;
                std::vector<char> chosen(g.size(), 0);
                for (size_t i = 0; i < quota; ++i) chosen[(size_t)(i * stride)] = 1;
                for (size_t i = 0; i < g.size(); ++i) (chosen[i] ? kept : rest).push_back(g[i]);
            }
        }
        if (kept.size() >= max_frontier) {
            pick = spread(kept, max_frontier);
        } else {
            pick = kept;
            if (!rest.empty()) {
                std::vector<size_t> extra = spread(rest, max_frontier - kept.size());
                pick.insert(pick.end(), extra.begin(), extra.end());
            }
        }
    }
    std::vector<Node> out;
    out.reserve(pick.size());
    for (size_t i : pick) out.push_back(std::move(frontier[i]));
    return out;
}

// search.find_path
SearchResult find_path(Level& L, double goal_x, int max_ticks, int input_period, int max_frontier, double time_limit,
                       bool use_bucket) {
    Simulator sim(L);
    auto t0 = std::chrono::steady_clock::now();
    Player init = sim.initial_state();
    std::vector<Node> frontier{{init, -1, make_key(init)}};
    std::vector<std::pair<int, bool>> history;
    double best_x = init.x;
    int best_node = -1, peak = 1;
    SearchResult res;
    auto inputs_of = [&](int node) {
        std::vector<uint8_t> out;
        while (node >= 0) { out.push_back(history[node].second); node = history[node].first; }
        std::reverse(out.begin(), out.end());
        return out;
    };
    for (int tick = 0; tick < max_ticks; ++tick) {
        bool branch = tick % input_period == 0;
        std::vector<Node> next;
        next.reserve(frontier.size() * 2);
        std::unordered_map<Key, size_t, KeyHash> seen;
        seen.reserve(frontier.size() * 4);
        for (const Node& n : frontier) {
            bool opts[2] = {false, true};
            int nopt = 2;
            if (!branch) { opts[0] = n.node >= 0 ? history[n.node].second : false; nopt = 1; }
            for (int oi = 0; oi < nopt; ++oi) {
                bool hold = opts[oi];
                Player nxt = sim.step(n.st, hold);
                if (nxt.dead) continue;
                Key k = make_key(nxt);
                if (seen.count(k)) continue;
                history.push_back({n.node, hold});
                seen.emplace(k, next.size());
                next.push_back({std::move(nxt), (int)history.size() - 1, std::move(k)});
            }
        }
        if (next.empty()) break;
        frontier.swap(next);
        if ((int)frontier.size() > max_frontier) frontier = prune(frontier, (size_t)max_frontier, use_bucket);
        peak = std::max(peak, (int)frontier.size());
        size_t lead = 0;
        for (size_t i = 1; i < frontier.size(); ++i)
            if (frontier[i].st.x > frontier[lead].st.x) lead = i;
        if (frontier[lead].st.x > best_x) { best_x = frontier[lead].st.x; best_node = frontier[lead].node; }
        if (best_x >= goal_x) {
            res.passed = true;
            res.max_x = best_x;
            res.ticks = tick + 1;
            res.inputs = inputs_of(best_node);
            res.frontier_peak = peak;
            return res;
        }
        if (time_limit > 0 && tick % 64 == 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > time_limit)
            break;
    }
    res.passed = false;
    res.max_x = best_x;
    res.inputs = inputs_of(best_node);
    res.ticks = (int)res.inputs.size();
    res.frontier_peak = peak;
    return res;
}

}  // namespace gd

// -------------------------------------------------------------------------------------------------- C API
using namespace gd;

#pragma pack(push, 8)
struct StateView {
    double x, y, prev_x, prev_y, y_vel, speed, robot_boost, gravity_mod, spider_tp_time, dash_slope, time,
        last_flip_time, last_mode_change_time, cor_min, cor_max, corridor_prev_max, pending_speed, slope_y_vel,
        slope_angle, slope_vs, slope_enter_time;
    int64_t tick;
    int32_t mode, mini, upside_down, held, buffer, ring_held_snap, ring_buf_snap, used_normal_ring, on_ground, grounded,
        is_rising, outside_clamp, jump_released, spider_pending, is_dashing, teleported, fix_gravity_bug, dead, killer,
        oob_prev, has_corridor, free_mode, has_pending_speed, prev_top_uid, prev_bottom_uid, on_slope, was_on_slope,
        slope_ceil, slope_uid, slope_desc, slope_clamped;
    int32_t n_touched, n_activated, n_speed_log;
    int32_t touched[32];
    int32_t activated[128];
    double speed_log[3 * 64];
};
struct SearchOut {
    int32_t passed, ticks, frontier_peak, n_inputs;
    double max_x;
};
#pragma pack(pop)

extern "C" {

int gd_abi_version() { return 2; }
int gd_obj_rec_size() { return (int)sizeof(ObjRec); }
int gd_state_view_size() { return (int)sizeof(StateView); }

void* gd_level_new(const ObjRec* objs, int n, const SettingsRec* s) {
    Level* L = new Level();
    L->objects.assign(objs, objs + n);
    L->settings = *s;
    L->build_index();
    return L;
}

void gd_level_set_engine(void* lvl, const TrigRec* trigs, int nt, const int32_t* trig_groups, const int32_t* trig_off,
                         const SetRec* sets, int ns, const int32_t* set_groups, const int32_t* set_group_off,
                         const int32_t* set_objs, const int32_t* set_obj_off) {
    Level* L = (Level*)lvl;
    L->triggers.clear();
    for (int i = 0; i < nt; ++i) {
        const TrigRec& r = trigs[i];
        Trigger t;
        t.kind = r.kind; t.target = r.target; t.spawn_triggered = r.spawn_triggered; t.lock_x = r.lock_x;
        t.enable = r.enable; t.easing = r.easing; t.x = r.x; t.dx = r.dx; t.dy = r.dy; t.duration = r.duration;
        t.rate = r.rate; t.mod_x = r.mod_x; t.delay = r.delay;
        t.groups.assign(trig_groups + trig_off[i], trig_groups + trig_off[i + 1]);
        L->triggers.push_back(std::move(t));
    }
    L->sets.clear();
    for (int i = 0; i < ns; ++i) {
        GroupSet g;
        g.groups.assign(set_groups + set_group_off[i], set_groups + set_group_off[i + 1]);
        g.objects.assign(set_objs + set_obj_off[i], set_objs + set_obj_off[i + 1]);
        g.lo = sets[i].lo;
        g.hi = sets[i].hi;
        g.always = sets[i].always;
        L->sets.push_back(std::move(g));
    }
    L->finish_engine();
}

void gd_level_free(void* lvl) { delete (Level*)lvl; }

void* gd_initial(void* lvl) {
    Simulator sim(*(Level*)lvl);
    return new Player(sim.initial_state());
}

void* gd_step(void* lvl, void* st, int hold) {
    Simulator sim(*(Level*)lvl);
    return new Player(sim.step(*(Player*)st, hold != 0));
}

void gd_state_free(void* st) { delete (Player*)st; }

int gd_state_view(void* lvl, void* st, StateView* v) {
    const Player& p = *(Player*)st;
    std::memset(v, 0, sizeof(StateView));
    v->x = p.x; v->y = p.y; v->prev_x = p.prev_x; v->prev_y = p.prev_y; v->y_vel = p.y_vel; v->speed = p.speed;
    v->robot_boost = p.robot_boost; v->gravity_mod = p.gravity_mod; v->spider_tp_time = p.spider_tp_time;
    v->dash_slope = p.dash_slope; v->time = p.time; v->last_flip_time = p.last_flip_time;
    v->last_mode_change_time = p.last_mode_change_time; v->cor_min = p.cor_min; v->cor_max = p.cor_max;
    v->corridor_prev_max = p.corridor_prev_max; v->pending_speed = p.pending_speed; v->tick = p.tick;
    v->mode = p.mode; v->mini = p.mini; v->upside_down = p.upside_down; v->held = p.held; v->buffer = p.buffer;
    v->ring_held_snap = p.ring_held_snap; v->ring_buf_snap = p.ring_buf_snap; v->used_normal_ring = p.used_normal_ring;
    v->on_ground = p.on_ground; v->grounded = p.grounded; v->is_rising = p.is_rising; v->outside_clamp = p.outside_clamp;
    v->jump_released = p.jump_released; v->spider_pending = p.spider_pending; v->is_dashing = p.is_dashing;
    v->teleported = p.teleported; v->fix_gravity_bug = p.fix_gravity_bug; v->dead = p.dead; v->killer = p.killer;
    v->oob_prev = p.oob_prev; v->has_corridor = p.has_corridor; v->free_mode = p.free_mode;
    v->has_pending_speed = p.has_pending_speed; v->prev_top_uid = p.prev_top_uid; v->prev_bottom_uid = p.prev_bottom_uid;
    v->slope_y_vel = p.slope_y_vel; v->slope_angle = p.slope_angle; v->slope_vs = p.slope_vs;
    v->slope_enter_time = p.slope_enter_time; v->on_slope = p.on_slope; v->was_on_slope = p.was_on_slope;
    v->slope_ceil = p.slope_ceil; v->slope_uid = p.slope_uid; v->slope_desc = p.slope_desc;
    v->slope_clamped = p.slope_clamped;
    v->n_touched = (int32_t)std::min<size_t>(p.touched_rings.size(), 32);
    for (int i = 0; i < v->n_touched; ++i) v->touched[i] = p.touched_rings[i];
    v->n_activated = (int32_t)std::min<size_t>(p.activated.size(), 128);
    for (int i = 0; i < v->n_activated; ++i) v->activated[i] = p.activated[i];
    std::vector<Break> log = ((Level*)lvl)->logs.materialize(p.speed_log);
    v->n_speed_log = (int32_t)std::min<size_t>(log.size(), 64);
    for (int i = 0; i < v->n_speed_log; ++i) {
        v->speed_log[3 * i] = (double)log[i].tick;
        v->speed_log[3 * i + 1] = log[i].x;
        v->speed_log[3 * i + 2] = log[i].v;
    }
    return (p.touched_rings.size() > 32 || p.activated.size() > 128 || log.size() > 64) ? 1 : 0;
}

int gd_find_path(void* lvl, double goal_x, int max_ticks, int input_period, int max_frontier, double time_limit,
                 int use_bucket, SearchOut* out, uint8_t* inputs, int inputs_cap) {
    SearchResult r = find_path(*(Level*)lvl, goal_x, max_ticks, input_period, max_frontier, time_limit, use_bucket != 0);
    out->passed = r.passed;
    out->ticks = r.ticks;
    out->frontier_peak = r.frontier_peak;
    out->max_x = r.max_x;
    out->n_inputs = (int32_t)r.inputs.size();
    int n = std::min<int>((int)r.inputs.size(), inputs_cap);
    if (inputs && n > 0) std::memcpy(inputs, r.inputs.data(), n);
    return r.inputs.size() > (size_t)inputs_cap ? 1 : 0;
}

int gd_unsupported(void* lvl, char* buf, int cap) {
    std::string s;
    for (const auto& kv : ((Level*)lvl)->unsupported) {
        char num[64];
        std::snprintf(num, sizeof num, "%.17g", kv.second);
        s += kv.first + "=" + num + ";";
    }
    int n = std::min<int>((int)s.size(), cap - 1);
    if (cap > 0) { std::memcpy(buf, s.data(), n); buf[n] = 0; }
    return (int)s.size();
}

}  // extern "C"
