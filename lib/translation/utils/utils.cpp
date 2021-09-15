#include "translation/utils/utils.h"

namespace translation {
namespace utils {

std::vector<std::string> splitString(const std::string &str, char delim) {
  std::vector<std::string> lines{};

  std::string::size_type pos;
  std::string::size_type prev = 0;
  while ((pos = str.find(delim, prev)) != std::string::npos) {
    lines.push_back(str.substr(prev, pos - prev));
    prev = pos + 1;
  }
  if (prev < str.length()) {
    lines.push_back(str.substr(prev));
  }
  return lines;
}

std::vector<std::string> getLines(const std::string &str) {
  return splitString(str, '\n');
}

std::string indentAllLines(const std::string &str) {

  std::vector<std::string> lines = getLines(str);

  std::string rv;
  for (const auto &line : lines) {
    rv += "\t" + line + "\n";
  }
  return rv;
}

bool isWhitespace(unsigned char c) {
  return (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
          c == '\f');
}

} // namespace utils
} // namespace murphi