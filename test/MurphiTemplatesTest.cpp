#include "translation/murphi/MurphiCodeGen.h"
#include "translation/utils/murphi-utils.h"
#include <gtest/gtest.h>

using namespace murphi;
using namespace murphi::utils;

TEST(MurphiTemplateTests, IndentLines) {
  std::string testString = "line one\nline two\nline three\n";

  std::string indentedString = indentAllLines(testString);

  std::string expstr = "\tline one\n\tline two\n\tline three\n";

  ASSERT_STREQ(expstr.c_str(), indentedString.c_str());
}

TEST(MurphiTemplateTests, IndentLinesWithoutTrailingEndl) {
  std::string testString = "line one \n line two \n line three";

  std::string indentedString = indentAllLines(testString);

  // If the line does not include a newline - a newline will be added
  std::string expstr = "\tline one \n\t line two \n\t line three\n";

  ASSERT_STREQ(expstr.c_str(), indentedString.c_str());
}

TEST(MurphiTemplateTests, InterleaveFunctions) {
  std::vector<std::string> elems{"hello", "world", "is", "a", "sentence"};
  auto interleaved = interleave<std::string>(elems, " ");
  ASSERT_STREQ(interleaved.c_str(), "hello world is a sentence");
}

TEST(MurphiCodeGen, ConstantTemplate) {
  std::string id = "NrCaches";
  size_t val = 3;

  MurphiConstantTemplate murConst(id, val);

  auto printedValue = murConst.to_string();

  ASSERT_STREQ("NrCaches : 3", printedValue.c_str());
}

TEST(MurphiCodeGen, PrintParametersListProcedureTemplate) {
  std::string param1 = "var a: Machine";
  std::string param2 = "var b: Cache";
  std::string param3 = "var c: Object";

  MurphiProcedureTemplate proc("handle_cache_evict");

  proc.addParameter(param1);
  proc.addParameter(param2);
  proc.addParameter(param3);

  auto printedValue = proc.printParametersList();

  ASSERT_STREQ("var a: Machine; var b: Cache; var c: Object",
               printedValue.c_str());
}

TEST(MurphiCodeGen, PrintForwardDeclarationsTemplate) {
  std::string fwdDecl1 = "var msg : Message;";
  std::string fwdDecl2 = "var msg2 : Message;";

  MurphiProcedureTemplate proc("handle_cache_evict");

  proc.addForwardDeclaration(fwdDecl1);
  proc.addForwardDeclaration(fwdDecl2);

  auto printedValue = proc.printForwardDeclarations();

  ASSERT_STREQ("var msg : Message;\nvar msg2 : Message;", printedValue.c_str());
}

TEST(MurphiCodeGen, PrintProcedure){
  std::string procName = "handle_cache_I_evict_GetM";
  std::string inMsgParam = "inmsg: Message";
  std::string cachesetParam = "cache: OBJSET_cache";
  std::string fwdMsgDecl = "var msg : Message;";
  std::string testBody = "cache.State := cache_I;";

  MurphiProcedureTemplate proc(procName);
  proc.addParameter(inMsgParam);
  proc.addParameter(cachesetParam);
  proc.addForwardDeclaration(fwdMsgDecl);
  proc.setProcedureBody(testBody);

  std::string printedValue = proc.to_string();

  std::string expectedStr = "procedure handle_cache_I_evict_GetM(inmsg: Message; cache: OBJSET_cache);\n"
                            "\tvar msg : Message;\n"
                            "begin\n"
                            "\tcache.State := cache_I;\n"
                            "end;\n";

  ASSERT_STREQ(expectedStr.c_str(), printedValue.c_str());
}

TEST(MurphiCodeGen, PrintProcedureWithAliases){
  std::string procName = "handle_cache_I_evict_GetM";
  std::string inMsgParam = "inmsg: Message";
  std::string cachesetParam = "cache: OBJSET_cache";
  std::string fwdMsgDecl = "var msg : Message;";
  std::string testBody = "cache.State := cache_I;";

  // create some aliases
  MurphiAliasStatement msgAlias("msg", "inmsg");
  MurphiAliasStatement cacheAlias("cle", "i_cache[m].CL[adr]");


  MurphiProcedureTemplate proc(procName);
  proc.addParameter(inMsgParam);
  proc.addParameter(cachesetParam);
  proc.addForwardDeclaration(fwdMsgDecl);
  proc.setProcedureBody(testBody);

  // add aliases to the procedure
  proc.addAliasStatement(msgAlias);
  proc.addAliasStatement(cacheAlias);

  std::string printedValue = proc.to_string();

  std::string expectedStr = "procedure handle_cache_I_evict_GetM(inmsg: Message; cache: OBJSET_cache);\n"
                            "\tvar msg : Message;\n"
                            "begin\n"
                            "\talias msg : inmsg do\n"
                            "\t\talias cle : i_cache[m].CL[adr] do\n"
                            "\t\t\tcache.State := cache_I;\n"
                            "\t\tendalias;\n"
                            "\tendalias;\n"
                            "end;\n";

  ASSERT_STREQ(expectedStr.c_str(), printedValue.c_str());
}
