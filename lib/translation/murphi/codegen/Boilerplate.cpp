#include "translation/murphi/codegen/Boilerplate.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"

using namespace mlir;
using namespace murphi;
using namespace inja;

namespace murphi {

/// BOILERPLATE ///
namespace boilerplate {

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

json getMessageHandlerSwitch() {
  return {};
//  auto returnFalseStmt =
//      murphi::detail::ReturnStmt{murphi::detail::ExprID{"false"}};
//  // switch inmsg.mtype
//  auto switchStmt = murphi::detail::SwitchStmt{
//      murphi::detail::Designator{
//          murphi::detail::c_inmsg,
//          {murphi::detail::Indexer{"object", murphi::detail::ExprID{""}}}},
//      {},
//      {}};
}

} // namespace boilerplate
} // namespace murphi