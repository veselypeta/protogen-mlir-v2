#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"

namespace murphi {
/*
 * Implementation Details
 */
namespace detail {
/*
 * ConstDecl to_json helper method
 */
void to_json(json &j, const ConstDecl &c) {
  j = {{"id", c.id}, {"value", c.value}};
}
/*
 * Enum to_json helper method
 */
void to_json(json &j, const Enum &c) {
  j = {{"id", c.id}, {"typeId", "enum"}, {"type", {{"decls", c.elems}}}};
}

/*
 * Network Decl helper
 */
json emitNetworkDefinitionJson() {
  auto ord_type_name = std::string(ObjKey) + ordered;
  auto ord_type_count_name = std::string(ObjKey) + ordered_cnt;
  auto un_ord_type_name = std::string(ObjKey) + unordered;
  json j = {
      {{"id", ord_type_name},
       {"typeId", "array"},
       {"type",
        {{"index", {{"typeId", "ID"}, {"type", k_machines}}},
         {"type",
          {{"typeId", "array"},
           {"type",
            {{"index",
              {{"typeId", "sub_range"},
               {"type",
                {{"start", 0}, {"stop", std::string(c_ordered) + "-1"}}}}},
             {"type", {{"typeId", "ID"}, {"type", r_message}}}}}

          }}}}},
      {{"id", ord_type_count_name},
       {"typeId", "array"},
       {"type",
        {{"index", {{"typeId", "ID"}, {"type", k_machines}}},
         {"type",
          {{"typeId", "sub_range"},
           {"type", {{"start", 0}, {"stop", c_unordered}}}

          }}}}},
      {{"id", un_ord_type_name},
       {"typeId", "array"},
       {"type",
        {{"index", {{"typeId", "ID"}, {"type", k_machines}}},
         {"type",
          {{"typeId", "multiset"},
           {"type",
            {{"index", {{"typeId", "ID"}, {"type", c_ordered}}},
             {"type", {{"typeId", "ID"}, {"type", r_message}}}}}}}}}}};
  return j;
}

bool validateMurphiJSON(const json &j) {
  const std::string schema_path =
      std::string(JSONValidation::schema_base_directory) +
      "gen_Murphi_json_schema.json";
  return JSONValidation::validate_json(schema_path, j);
}
} // namespace detail

mlir::LogicalResult MurphiCodeGen::translate() {
  generateConstants();
  generateTypes();
  return render();
}

void MurphiCodeGen::generateConstants() {
  // add the defined constants
  for (auto constantOp : moduleInterpreter.getConstants()) {
    data["decls"]["const_decls"].push_back(
        detail::ConstDecl{constantOp.id().str(), constantOp.val()});
  }

  // boilerplate constants
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_val_cnt_id, detail::c_val_max});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_adr_cnt_id, detail::c_adr_cnt});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_ordered, detail::c_ordered_size});
  data["decls"]["const_-decls"].push_back(
      detail::ConstDecl{detail::c_unordered, detail::c_unordered_size});
}

void MurphiCodeGen::generateTypes() {
  _typeEnums();
  _typeStatics();
  _typeMachines();
  // TODO - type machines
  _typeMessage();
  _typeNetworkObjects();
}

mlir::LogicalResult MurphiCodeGen::render() {
  // validate json
//  assert(detail::validateMurphiJSON(data) &&
//         "JSON from codegen does not validate with the json schema");
  auto &env = InjaEnvSingleton::getInstance();
  auto tmpl = env.parse_template("murphi_base.tmpl");
  auto result = env.render(tmpl, data);
  output << result;
  return mlir::success();
}

[[nodiscard]] static std::string k_directory_state_t() {
  return detail::machines.directory.str() + detail::state_suffix;
}

[[nodiscard]] static std::string k_cache_state_t() {
  return detail::machines.cache.str() + detail::state_suffix;
}

void MurphiCodeGen::_typeEnums() {
  // access enum
  data["decls"]["type_decls"].push_back(
      detail::Enum{detail::k_access, {"none", "load", "store"}});
  // messages enum
  data["decls"]["type_decls"].push_back(detail::Enum{
      detail::k_message_type, moduleInterpreter.getEnumMessageTypes()});
  // cache state
  data["decls"]["type_decls"].push_back(detail::Enum{
      k_cache_state_t(),
      moduleInterpreter.getEnumMachineStates(detail::machines.cache.str())});
  data["decls"]["type_decls"].push_back(detail::Enum{
      k_directory_state_t(), moduleInterpreter.getEnumMachineStates(
                                 detail::machines.directory.str())});
}

void MurphiCodeGen::_typeStatics() {
  // Address type
  data["decls"]["type_decls"].push_back(
      {{"id", detail::k_address},
       {"typeId", "scalarset"},
       {"type", {{"type", detail::c_adr_cnt_id}}}});
  // ClValue type
  data["decls"]["type_decls"].push_back(
      {{"id", detail::k_cache_val},
       {"typeId", "sub_range"},
       {"type", {{"start", 0}, {"stop", detail::c_val_cnt_id}}}});
}

void MurphiCodeGen::_typeNetworkObjects() {
  for (const auto &type : detail::emitNetworkDefinitionJson()) {
    data["decls"]["type_decls"].push_back(type);
  }
}

void MurphiCodeGen::_typeMachines() {
  // *** cache *** //
  mlir::pcc::CacheDeclOp cacheOp = moduleInterpreter.getCache();
  _getMachineEntry(cacheOp.getOperation());

  // *** directory *** //
  //    mlir::pcc::DirectoryDeclOp directoryOp =
  //    moduleInterpreter.getDirectory();
}

void MurphiCodeGen::_getMachineEntry(mlir::Operation *machineOp) {
  // Check that the operation is either cache or directory decl
  auto opIdent = machineOp->getName().getIdentifier().strref();
  assert(opIdent == detail::opStringMap.cache_decl ||
         opIdent == detail::opStringMap.dir_decl &&
             "invalid operation passed to generation machine function");
  auto machineIdent = opIdent == detail::opStringMap.cache_decl
                          ? detail::machines.cache
                          : detail::machines.directory;
  json r_cache;
  for (auto &attr : machineOp->getAttrs()) {
    if (attr.first != "id") {
      std::string fieldID = attr.first.str();
      mlir::TypeAttr typeAttr = attr.second.cast<mlir::TypeAttr>();
      std::string fieldType =
          MLIRTypeToMurphiTypeRef(typeAttr.getValue(), machineIdent);
      json entry = {{"id", fieldID}, {"typeId", "ID"}, {"type", fieldType}};
      r_cache["decls"].push_back(entry);
    }
  }
  data["decls"]["type_decls"].push_back(
      {{"id", std::string(detail::EntryKey) + detail::machines.cache.str()},
       {"typeId", "record"},
       {"type", r_cache}});
}

void MurphiCodeGen::_typeMessage() {
  json msgJson = {
      {"id", detail::r_message},
      {"typeId", "record"},
      {"type", {{"decls", json::array() /* initialize an empty array */}}}};
  // default types
  for (auto &defMsgType : detail::BaseMsg) {
    msgJson["type"]["decls"].push_back({{"id", defMsgType.first},
                                        {"typeId", "ID"},
                                        {"type", defMsgType.second}});
  }
  // extra types
  for (auto &adiType : detail::SuperMsg) {
    msgJson["type"]["decls"].push_back(
        {{"id", adiType.first}, {"typeId", "ID"}, {"type", adiType.second}});
  }
  data["decls"]["type_decls"].push_back(msgJson);
}

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t,
                                    const llvm::StringRef mach) {
  if (t.isa<mlir::pcc::DataType>()) {
    return detail::k_cache_val;
  }
  if (t.isa<mlir::pcc::StateType>()) {
    return mach == detail::machines.cache ? k_cache_state_t()
                                          : k_directory_state_t();
  }
  // TODO - add support for more types
  assert(0 && "currently using an unsupported type!");
}
} // namespace murphi
