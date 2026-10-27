@echo off
rem Builds the gdgen replay mod with MSVC; the .geode package ends up in build\.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
rem Geode SDK / bindings / CLI default to <repo>\geode\ (not part of this repo); set the variables to override.
if not defined GEODE_SDK set GEODE_SDK=%~dp0..\geode\sdk
if not defined GEODE_BINDINGS_REPO_PATH set GEODE_BINDINGS_REPO_PATH=%~dp0..\geode\bindings
set PATH=%~dp0..\geode\cli;%PATH%
cd /d %~dp0
cmake -B build -G "Visual Studio 17 2022" -A x64 || exit /b 1
cmake --build build --config RelWithDebInfo || exit /b 1
