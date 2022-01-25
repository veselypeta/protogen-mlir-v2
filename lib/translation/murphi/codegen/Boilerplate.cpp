#include "translation/murphi/codegen/Boilerplate.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/murphi/codegen/MurphiConstants.h"

using namespace murphi;
using namespace murphi::detail;
using namespace inja;

namespace murphi {

/// BOILERPLATE ///
namespace boilerplate {

/*
 * Network Decl helper
 */
json emitNetworkDefinitionJson() {
  auto ord_type_name = std::string(ObjKey) + ordered;
  auto ord_type_count_name = std::string(ObjKey) + ordered_cnt;
  auto un_ord_type_name = std::string(ObjKey) + unordered;
  detail::TypeDecl<detail::Array<
      detail::ID,
      detail::Array<detail::SubRange<size_t, std::string>, detail::ID>>>
      ordered_network_type{
          ord_type_name,
          {{e_machines_t},
           {{0, std::string(c_ordered_t) + "-1"}, {r_message_t}}}};
  detail::TypeDecl<
      detail::Array<detail::ID, detail::SubRange<size_t, std::string>>>
      ordered_network_count{ord_type_name, {{e_machines_t}, {0, c_ordered_t}}};
  detail::TypeDecl<
      detail::Array<detail::ID, detail::Multiset<detail::ID, detail::ID>>>
      unordered_network_t{un_ord_type_name,
                          {{e_machines_t}, {{c_unordered_t}, {r_message_t}}}};
  json j = {ordered_network_type, ordered_network_count, unordered_network_t};
  return j;
}

// Constants
void setBoilerplateConstants(json &data) {
  // boilerplate constants
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_val_cnt_t, detail::c_val_max});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_adr_cnt_t, detail::c_adr_cnt});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_ordered_t, detail::c_ordered_size});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_unordered_t, detail::c_unordered_size});
}

void setBoilerplateAccessType(json &data) {
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_access_t, detail::Enum{{"none", "load", "store"}}});
}

SwitchStmt getMessageHandlerSwitch() {
  auto returnFalseStmt =
      murphi::detail::ReturnStmt{murphi::detail::ExprID{"false"}};
  // switch inmsg.mtype
  return murphi::detail::SwitchStmt{
      murphi::detail::Designator{
          murphi::detail::c_inmsg,
          {murphi::detail::Indexer{
              "object", murphi::detail::ExprID{murphi::detail::c_mtype}}}},
      {},
      {std::move(returnFalseStmt)}};
}

SwitchStmt getStateHandlerSwitch() {
  // switch cle.State
  auto switchStmt = murphi::detail::SwitchStmt{
      murphi::detail::Designator{
          murphi::detail::cle_a,
          {murphi::detail::Indexer{
              "object", murphi::detail::ExprID{murphi::detail::c_state}}}},
      {},
      {}};
  return switchStmt;
}

CaseStmt getBasicCaseStmt(const std::string &caseText){
  return CaseStmt{
      ExprID{caseText},
      {}
  };
}

CaseStmt getBasicCaseStmt(std::string&& caseText){
  return CaseStmt{
      ExprID{std::move(caseText)},
      {}
  };
}

} // namespace boilerplate
} // namespace murphi