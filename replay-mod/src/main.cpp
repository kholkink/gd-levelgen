// gdgen replay: plays input sequences tick by tick in the real game and logs what happens.
//
// Reads <dir>/job.txt (dir = %GDGEN_REPLAY_DIR% or D:/gd-levelgen/replay) once at the main menu:
//   time_scale <float>                  speed-up via the cocos scheduler (physics stays 240 Hz per step)
//   quit_when_done <0|1>
//   job <name> <level_file> <song_id> <audio_track> <max_ticks> <trace 0|1>
//   inputs <tick>:<1|0> ...             button 1 press/release, applied at the start of that physics step
// level_file holds the level string as stored on the servers (base64 gzip), relative to <dir>.
// Writes <dir>/out/results.txt (one line per job) and <dir>/out/<name>.csv (per-step player state).
// job.txt is renamed to job.running.txt (job.done.txt when finished); a new job.txt is picked up
// automatically while the game runs.
//
// Also imports generated levels: every <dir>/import/<name>.txt (a level string) becomes a new level in
// Create -> My Levels named <name>, and the file is renamed to <name>.imported (scripts/watch_levels.py
// drops the levels there as soon as they are generated).

#include <Geode/Geode.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>

using namespace geode::prelude;
namespace fs = std::filesystem;

namespace {

struct Job {
    std::string name;
    std::string levelFile;
    int songId = 0;
    int audioTrack = 0;
    int maxTicks = 200000;
    bool trace = true;
    std::vector<std::pair<int, bool>> inputs;
};

struct State {
    std::vector<Job> jobs;
    size_t next = 0;
    size_t current = 0;
    bool batchActive = false;
    bool running = false;
    bool finished = false;
    bool quitWhenDone = false;
    float timeScale = 1.f;
    int tick = 0;
    int halfTicks = 0;
    size_t inputPos = 0;
    std::ofstream trace;
} g;

fs::path baseDir() {
    if (auto env = std::getenv("GDGEN_REPLAY_DIR")) return fs::path(env);
    return fs::path("D:/gd-levelgen/replay");
}

void appendResult(std::string const& line) {
    std::ofstream out(baseDir() / "out" / "results.txt", std::ios::app);
    out << line << "\n";
}

std::string readFile(fs::path const& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

bool loadJobs() {
    auto path = baseDir() / "job.txt";
    std::ifstream in(path);
    if (!in) return false;
    g.jobs.clear();
    g.next = g.current = 0;
    g.timeScale = 1.f;
    g.quitWhenDone = false;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        if (kw == "time_scale") {
            ss >> g.timeScale;
        } else if (kw == "quit_when_done") {
            int q = 0;
            ss >> q;
            g.quitWhenDone = q != 0;
        } else if (kw == "job") {
            Job job;
            int trace = 1;
            ss >> job.name >> job.levelFile >> job.songId >> job.audioTrack >> job.maxTicks >> trace;
            job.trace = trace != 0;
            g.jobs.push_back(std::move(job));
        } else if (kw == "inputs" && !g.jobs.empty()) {
            std::string tok;
            while (ss >> tok) {
                auto colon = tok.find(':');
                if (colon == std::string::npos) continue;
                g.jobs.back().inputs.emplace_back(std::stoi(tok.substr(0, colon)), tok.substr(colon + 1) == "1");
            }
        }
    }
    in.close();
    std::error_code ec;
    fs::create_directories(baseDir() / "out", ec);
    fs::rename(path, baseDir() / "job.running.txt", ec);
    return !g.jobs.empty();
}

void startNextJob();

void finishJob(std::string const& status, std::string const& detail) {
    if (g.finished) return;
    g.finished = true;
    g.running = false;
    appendResult(std::format("{} status={} tick={} half_ticks={} {}",
        g.jobs[g.current].name, status, g.tick, g.halfTicks, detail));
    if (g.trace.is_open()) g.trace.close();
    geode::queueInMainThread([] {
        if (auto pl = PlayLayer::get()) pl->onQuit();
        startNextJob();
    });
}

void startNextJob() {
    if (PlayLayer::get()) {
        // onQuit tears the level down over a transition; try again next frame.
        geode::queueInMainThread([] { startNextJob(); });
        return;
    }
    auto director = CCDirector::sharedDirector();
    if (g.next >= g.jobs.size()) {
        director->getScheduler()->setTimeScale(1.f);
        appendResult("# all jobs done");
        g.batchActive = false;
        std::error_code ec;
        fs::rename(baseDir() / "job.running.txt", baseDir() / "job.done.txt", ec);
        if (g.quitWhenDone) geode::utils::game::exit(false);
        return;
    }
    g.current = g.next++;
    auto& job = g.jobs[g.current];
    auto level = GJGameLevel::create();
    level->m_levelString = readFile(baseDir() / job.levelFile);
    level->m_levelName = job.name;
    level->m_songID = job.songId;
    level->m_audioTrack = job.audioTrack;
    level->m_levelType = GJLevelType::Editor;
    g.tick = 0;
    g.halfTicks = 0;
    g.inputPos = 0;
    g.finished = false;
    g.running = true;
    if (job.trace) {
        g.trace.open(baseDir() / "out" / (job.name + ".csv"), std::ios::trunc);
        g.trace << "tick,half,x,y,y_vel,ship,ufo,ball,wave,robot,spider,swing,upside,size,speed,on_ground,dead\n";
    }
    log::info("gdgen replay: starting job {} ({} inputs)", job.name, job.inputs.size());
    director->getScheduler()->setTimeScale(g.timeScale);
    director->replaceScene(PlayLayer::scene(level, false, false));
}

void writeTrace(PlayerObject* p, bool isHalfTick) {
    if (!g.trace.is_open() || !p) return;
    auto pos = p->getPosition();
    g.trace << std::format("{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{},{},{},{:.2f},{:.2f},{},{}\n",
        g.tick, isHalfTick ? 1 : 0, pos.x, pos.y, p->m_yVelocity,
        p->m_isShip ? 1 : 0, p->m_isBird ? 1 : 0, p->m_isBall ? 1 : 0, p->m_isDart ? 1 : 0,
        p->m_isRobot ? 1 : 0, p->m_isSpider ? 1 : 0, p->m_isSwing ? 1 : 0, p->m_isUpsideDown ? 1 : 0,
        p->m_vehicleSize, p->m_playerSpeed, p->m_isOnGround ? 1 : 0, p->m_isDead ? 1 : 0);
}

}  // namespace

// Polls for <dir>/job.txt every two seconds while no batch is running, so batches can be queued without
// restarting the game. Hooked into the scheduler's own update: a scheduled selector got dropped by the game
// after the first batch.
void pollJobs() {
    if (g.batchActive || !fs::exists(baseDir() / "job.txt")) return;
    if (loadJobs()) {
        g.batchActive = true;
        log::info("gdgen replay: {} jobs loaded", g.jobs.size());
        startNextJob();
    }
}

// Generated levels dropped into <dir>/import/ become local levels (Create -> My Levels).
void pollImports() {
    std::error_code ec;
    auto dir = baseDir() / "import";
    if (!fs::is_directory(dir, ec)) return;
    for (auto const& entry : fs::directory_iterator(dir, ec)) {
        auto path = entry.path();
        if (path.extension() != ".txt") continue;
        auto data = readFile(path);
        if (data.empty()) continue;
        auto name = path.stem().string();
        auto level = GameLevelManager::get()->createNewLevel();
        level->m_levelString = data;
        level->m_levelName = name;
        level->m_levelDesc = "generated by gdgen";
        fs::rename(path, fs::path(path).replace_extension(".imported"), ec);
        log::info("gdgen: imported {} ({} chars)", name, data.size());
        Notification::create(std::format("gdgen: \"{}\" added to My Levels", name), NotificationIcon::Success, 5.f)
            ->show();
    }
}

class $modify(ReplayScheduler, CCScheduler) {
    void update(float dt) {
        CCScheduler::update(dt);
        static float sinceCheck = 0.f;
        static bool menuSeen = false;
        if (!menuSeen) {
            // Wait for the main menu before the first batch (the game is still loading before that).
            auto scene = CCDirector::sharedDirector()->getRunningScene();
            menuSeen = scene && scene->getChildByType<MenuLayer>(0);
            if (!menuSeen) return;
        }
        sinceCheck += dt;
        if (sinceCheck < 2.f) return;
        sinceCheck = 0.f;
        pollJobs();
        if (!g.batchActive) pollImports();
    }
};

class $modify(ReplayGameLayer, GJBaseGameLayer) {
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        auto pl = PlayLayer::get();
        bool active = g.running && !g.finished && pl && static_cast<GJBaseGameLayer*>(pl) == this;
        if (active) {
            if (isHalfTick) g.halfTicks++;
            writeTrace(m_player1, isHalfTick);   // state after the previous step
            auto& inputs = g.jobs[g.current].inputs;
            while (g.inputPos < inputs.size() && inputs[g.inputPos].first <= g.tick) {
                this->handleButton(inputs[g.inputPos].second, 1, true);
                g.inputPos++;
            }
        }
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        if (active) {
            g.tick++;
            if (g.tick > g.jobs[g.current].maxTicks) {
                finishJob("timeout", std::format("x={:.3f}", m_player1->getPositionX()));
            }
        }
    }
};

class $modify(ReplayPlayLayer, PlayLayer) {
    void resetLevel() {
        PlayLayer::resetLevel();
        if (g.running && !g.finished) {
            g.tick = 0;
            g.halfTicks = 0;
            g.inputPos = 0;
        }
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        if (g.running && !g.finished && player == m_player1 && object != m_anticheatSpike) {
            auto pos = player->getPosition();
            std::string killer = object
                ? std::format("killer_id={} killer_x={:.2f} killer_y={:.2f}", object->m_objectID,
                    object->getPositionX(), object->getPositionY())
                : std::string("killer_id=-1");
            finishJob("died", std::format("x={:.3f} y={:.3f} {}", pos.x, pos.y, killer));
        }
        PlayLayer::destroyPlayer(player, object);
    }

    void levelComplete() {
        if (g.running && !g.finished) {
            finishJob("complete", std::format("x={:.3f}", m_player1->getPositionX()));
        }
        PlayLayer::levelComplete();
    }
};
