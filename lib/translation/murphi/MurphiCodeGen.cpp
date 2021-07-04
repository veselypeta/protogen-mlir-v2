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
                             "{3}"                   // proc body
                             "end;\n";               // end proc

  std::string aliasTemplate = "{0}{1}{2}";

  // for each alias indent -- in reverse order
  for(auto alias = aliasStatements.rbegin(); alias != aliasStatements.rend(); ++alias){
    procedureBody = fmt::format(aliasTemplate,
                                (*alias).begin_alias_to_string(),
                                murphi::utils::indentAllLines(procedureBody),
                                (*alias).end_alias_to_string());
  }

  return fmt::format(procTemplate, procedureName, printParametersList(),
                     murphi::utils::indentAllLines(printForwardDeclarations()),
                     murphi::utils::indentAllLines(procedureBody)
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
