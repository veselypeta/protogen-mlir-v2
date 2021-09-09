#include "translation/utils/murphi-utils.h"
#include "translation/murphi/MurphiCodeGen.h"

namespace murphi {
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

murphi::MurphiProcedureTemplate
make_cache_message_handler_procedure(std::string &procIdent) {
  murphi::MurphiProcedureTemplate proc(procIdent);
  const std::string defaultInMsgParam = "inmsg : Message";
  const std::string defaultMachineParam = "cache : OBJSET_cache";
  const std::string defaultMsgFwdDecl = "var msg : Message;";

  // add the parameters
  proc.addParameter(defaultInMsgParam);
  proc.addParameter(defaultMachineParam);

  // add the fwd decl
  proc.addForwardDeclaration(defaultMsgFwdDecl);

  // TODO - setup the aliases for the function
  // TODO - have an easy entrypoint for the consumer to add statements

  return proc;
}

bool isWhitespace(unsigned char c) {
  return (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
          c == '\f');
}

} // namespace utils
} // namespace murphi