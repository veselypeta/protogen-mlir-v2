#include "translation/murphi/MurphiCodeGen.h"
#include "fmt/core.h"
#include "translation/utils/murphi-utils.h"

namespace murphi {

std::string MurphiConstantTemplate::to_string() {
  return fmt::format("{0} : {1}", id, val);
}

std::string MurphiProcedureTemplate::printParametersList() {
  return murphi::utils::interleave<std::string>(parameterList, "; ");
}

std::string MurphiProcedureTemplate::printForwardDeclarations() {
  return murphi::utils::interleave<std::string>(forwardDeclarations, "\n");
}

std::string MurphiProcedureTemplate::to_string() {
  std::string procTemplate = "procedure {0}({1});\n" // proc name and parameters
                             "{2}"                   // fwd declarations
                             "begin\n"               // begin proc
                             "{3}"                   // begin alias
                             "{4}"                   // proc body
                             "{5}"                   // end alias
                             "end;\n";               // end proc

  return fmt::format(procTemplate, procedureName, printParametersList(),
                     murphi::utils::indentAllLines(printForwardDeclarations()),
                     /* begin aliases */ "",
                     murphi::utils::indentAllLines(procedureBody),
                     /* end aliases */ ""
                     );
}


std::string MurphiAliasStatement::begin_alias_to_string() {
  std::string aliasTemplate = "alias {0} : {1} do\n";
  return fmt::format(aliasTemplate, alias, typeAlias);
}

std::string MurphiAliasStatement::end_alias_to_string() {
  std::string endAliasTemplate = "endalias;\n";
  return endAliasTemplate;
}
} // namespace murphi
