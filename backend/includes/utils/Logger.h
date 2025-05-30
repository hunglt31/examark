#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

class Logger {
public:
    enum LogLevel {
        DEBUG,
        INFO,
        SUCCESS,
        WARNING,
        ERROR,
        CRITICAL
    };

    /**
     * @brief Log a message with the specified level and category
     * 
     * @param level Severity level of the message
     * @param category Category/component generating the log
     * @param message The message to log
     */
    static void log(LogLevel level, const std::string& category, const std::string& message);

    /**
     * @brief Log debug information
     */
    static void debug(const std::string& category, const std::string& message);
    
    /**
     * @brief Log general information
     */
    static void info(const std::string& category, const std::string& message);
    
    /**
     * @brief Log successful operations
     */
    static void success(const std::string& category, const std::string& message);
    
    /**
     * @brief Log warnings
     */
    static void warning(const std::string& category, const std::string& message);
    
    /**
     * @brief Log errors
     */
    static void error(const std::string& category, const std::string& message);
    
    /**
     * @brief Log critical errors
     */
    static void critical(const std::string& category, const std::string& message);

private:
    static std::string getCurrentTimestamp();
    static std::string getColorCode(LogLevel level);
    static std::string getLevelString(LogLevel level);
};