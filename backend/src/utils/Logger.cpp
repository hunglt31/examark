#include "utils/Logger.h"

// ANSI color codes
const std::string RESET = "\033[0m";
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";
const std::string BOLD = "\033[1m";

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string Logger::getColorCode(LogLevel level) {
    switch (level) {
        case DEBUG:   return CYAN;
        case INFO:    return BLUE;
        case SUCCESS: return GREEN;
        case WARNING: return YELLOW;
        case ERROR:   return RED;
        case CRITICAL: return BOLD + RED;
        default:      return RESET;
    }
}

std::string Logger::getLevelString(LogLevel level) {
    switch (level) {
        case DEBUG:    return "DEBUG";
        case INFO:     return "INFO";
        case SUCCESS:  return "SUCCESS";
        case WARNING:  return "WARNING";
        case ERROR:    return "ERROR";
        case CRITICAL: return "CRITICAL";
        default:       return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& category, const std::string& message) {
    std::string color = getColorCode(level);
    std::string levelStr = getLevelString(level);
    std::string timestamp = getCurrentTimestamp();
    
    std::cout << timestamp << " " 
              << color << "[" << levelStr << "]" << RESET << " "
              << MAGENTA << "[" << category << "]" << RESET << " " 
              << message << std::endl;
}

void Logger::debug(const std::string& category, const std::string& message) {
    log(DEBUG, category, message);
}

void Logger::info(const std::string& category, const std::string& message) {
    log(INFO, category, message);
}

void Logger::success(const std::string& category, const std::string& message) {
    log(SUCCESS, category, message);
}

void Logger::warning(const std::string& category, const std::string& message) {
    log(WARNING, category, message);
}

void Logger::error(const std::string& category, const std::string& message) {
    log(ERROR, category, message);
}

void Logger::critical(const std::string& category, const std::string& message) {
    log(CRITICAL, category, message);
}