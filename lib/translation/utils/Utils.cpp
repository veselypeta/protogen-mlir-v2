#include "translation/utils/utils.h"
#include "translation/murphi/codegen/MurphiConstants.h"
#include <cassert>

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

std::string indentAllLines(const std::string &str, size_t level) {

  std::vector<std::string> lines = getLines(str);

  std::string rv;
  std::string indentation = std::string(level*4, ' ');
  for (const auto &line : lines) {
    if(&line == &lines.back()){
      rv += indentation + line;
    } else {
      rv += indentation + line + "\n";
    }
  }
  return rv;
}

bool isWhitespace(unsigned char c) {
  return (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
          c == '\f');
}

std::string mangleCacheState(const std::string &state){
  return mangleState(state, murphi::detail::cache_state_prefix);
}


std::string mangleDirectoryState(const std::string &state){
  return mangleState(state, murphi::detail::directory_state_prefix);
}

std::string mangleState(const std::string &state, const std::string &prefix){
  return prefix + state;
}

std::string demangleState(const std::string &state){
  auto split = state.find( '_');
  assert(split != std::string::npos && "Could not demangle state name");
  return state.substr(split+1, state.length());
}

} // namespace utils
} // namespace murphi