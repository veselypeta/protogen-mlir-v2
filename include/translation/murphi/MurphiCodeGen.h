#pragma once
#include <string>
#include <vector>

namespace murphi {

class MurphiConstantTemplate {
public:
  explicit MurphiConstantTemplate(std::string id, unsigned long val)
      : id{std::move(id)}, val{val} {}

  std::string to_string();

private:
  std::string id;
  unsigned long val;
};

class MurphiAliasStatement {
public:
  MurphiAliasStatement(std::string alias, std::string typeAlias)
      : alias{std::move(alias)}, typeAlias{std::move(typeAlias)} {}

  std::string begin_alias_to_string();
  std::string end_alias_to_string();

private:
  std::string alias;
  std::string typeAlias;
};

class MurphiProcedureTemplate {
public:
  explicit MurphiProcedureTemplate(std::string procIdent)
      : procedureName{std::move(procIdent)} {}

  void addParameter(const std::string &val) { parameterList.push_back(val); }
  std::string printParametersList();
  void addForwardDeclaration(const std::string &val) {
    forwardDeclarations.push_back(val);
  }
  std::string printForwardDeclarations();

  // TODO - change to make it easier to add statements to the function
  void setProcedureBody(std::string &text) { procedureBody = text; }
  std::string &getProcedureBody() {return procedureBody;}

  void addAliasStatement(MurphiAliasStatement &alias){
    aliasStatements.push_back(alias);
  }

  std::string to_string();

private:
  std::string procedureName;
  std::vector<std::string> parameterList;
  std::vector<std::string> forwardDeclarations;
  std::vector<MurphiAliasStatement> aliasStatements;
  std::string procedureBody;
};
} // namespace murphi
