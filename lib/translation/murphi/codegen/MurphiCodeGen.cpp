#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <cstdarg>


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
 * Union to_json helper method
 */
void to_json(json &j, const Union &u) {
  j = {{"id", u.id}, {"typeId", "union"}, {"type", {{"listElems", u.elems}}}};
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
        {{"index", {{"typeId", "ID"}, {"type", e_machines_t}}},
         {"type",
          {{"typeId", "array"},
           {"type",
            {{"index",
              {{"typeId", "sub_range"},
               {"type",
                {{"start", 0}, {"stop", std::string(c_ordered_t) + "-1"}}}}},
             {"type", {{"typeId", "ID"}, {"type", r_message_t}}}}}

          }}}}},
      {{"id", ord_type_count_name},
       {"typeId", "array"},
       {"type",
        {{"index", {{"typeId", "ID"}, {"type", e_machines_t}}},
         {"type",
          {{"typeId", "sub_range"},
           {"type", {{"start", 0}, {"stop", c_unordered_t}}}

          }}}}},
      {{"id", un_ord_type_name},
       {"typeId", "array"},
       {"type",
        {{"index", {{"typeId", "ID"}, {"type", e_machines_t}}},
         {"type",
          {{"typeId", "multiset"},
           {"type",
            {{"index", {{"typeId", "ID"}, {"type", c_ordered_t}}},
             {"type", {{"typeId", "ID"}, {"type", r_message_t}}}}}}}}}}};
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

bool MurphiCodeGen::is_json_valid() { return detail::validateMurphiJSON(data); }

void MurphiCodeGen::generateConstants() {
  // add the defined constants
  for (auto constantOp : moduleInterpreter.getConstants()) {
    data["decls"]["const_decls"].push_back(
        detail::ConstDecl{constantOp.id().str(), constantOp.val()});
  }

  // boilerplate constants
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_val_cnt_t, detail::c_val_max});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_adr_cnt_t, detail::c_adr_cnt});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_ordered_t, detail::c_ordered_size});
  data["decls"]["const_-decls"].push_back(
      detail::ConstDecl{detail::c_unordered_t, detail::c_unordered_size});
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

[[nodiscard]] static std::string e_directory_state_t() {
  return detail::machines.directory.str() + detail::state_suffix;
}

[[nodiscard]] static std::string e_cache_state_t() {
  return detail::machines.cache.str() + detail::state_suffix;
}

void MurphiCodeGen::_typeEnums() {
  // access enum
  data["decls"]["type_decls"].push_back(
      detail::Enum{detail::e_access_t, {"none", "load", "store"}});
  // messages enum
  data["decls"]["type_decls"].push_back(detail::Enum{
      detail::e_message_type_t, moduleInterpreter.getEnumMessageTypes()});
  // cache state
  data["decls"]["type_decls"].push_back(detail::Enum{
      e_cache_state_t(),
      moduleInterpreter.getEnumMachineStates(detail::machines.cache.str())});
  data["decls"]["type_decls"].push_back(detail::Enum{
      e_directory_state_t(), moduleInterpreter.getEnumMachineStates(
                                 detail::machines.directory.str())});
}

void MurphiCodeGen::_typeStatics() {
  // Address type
  data["decls"]["type_decls"].push_back(
      {{"id", detail::ss_address_t},
       {"typeId", "scalarset"},
       {"type", {{"type", detail::c_adr_cnt_t}}}});
  // ClValue type
  data["decls"]["type_decls"].push_back(
      {{"id", detail::sr_cache_val_t},
       {"typeId", "sub_range"},
       {"type", {{"start", 0}, {"stop", detail::c_val_cnt_t}}}});
}

void MurphiCodeGen::_typeNetworkObjects() {
  for (const auto &type : detail::emitNetworkDefinitionJson()) {
    data["decls"]["type_decls"].push_back(type);
  }
}

void MurphiCodeGen::_typeMachines() {
  // *** cache *** //
  mlir::pcc::CacheDeclOp cacheOp = moduleInterpreter.getCache();
  std::string cache_mach = _getMachineEntry(cacheOp.getOperation());

  // *** directory *** //
  mlir::pcc::DirectoryDeclOp directoryOp = moduleInterpreter.getDirectory();
  std::string dir_mach = _getMachineEntry(directoryOp.getOperation());

  _typeMachinesUnion({cache_mach, dir_mach});
}

std::string MurphiCodeGen::_getMachineEntry(mlir::Operation *machineOp) {
  // Check that the operation is either cache or directory decl
  const auto opIdent = machineOp->getName().getIdentifier().strref();
  assert(opIdent == detail::opStringMap.cache_decl ||
         opIdent == detail::opStringMap.dir_decl &&
             "invalid operation passed to generation machine function");
  const auto machineIdent = opIdent == detail::opStringMap.cache_decl
                                ? detail::machines.cache
                                : detail::machines.directory;
  json r_cache;

  const auto is_id_attr = [](const mlir::NamedAttribute &attr) {
    return attr.first == "id";
  };

  const auto generate_mach_attr_field =
      [&r_cache, &machineIdent](const mlir::NamedAttribute &attr) {
        std::string fieldID = attr.first.str();
        mlir::TypeAttr typeAttr = attr.second.cast<mlir::TypeAttr>();
        std::string fieldType =
            MLIRTypeToMurphiTypeRef(typeAttr.getValue(), machineIdent);
        json entry = {{"id", fieldID}, {"typeId", "ID"}, {"type", fieldType}};
        r_cache["decls"].push_back(entry);
      };

  // for each attribute; generate a machine attribute
  std::for_each(machineOp->getAttrs().begin(), machineOp->getAttrs().end(),
                [&](const mlir::NamedAttribute &attribute) {
                  const mlir::NamedAttribute named_attr =
                      static_cast<mlir::NamedAttribute>(attribute);
                  if (!is_id_attr(named_attr)) {
                    generate_mach_attr_field(named_attr);
                  }
                });

  std::string machine_id_t = std::string(detail::EntryKey) + machineIdent.str();
  // generate the corret murphi declaration
  data["decls"]["type_decls"].push_back(
      {{"id", machine_id_t },
       {"typeId", "record"},
       {"type", r_cache}});
  return machine_id_t;
}

void MurphiCodeGen::_typeMachinesUnion(const std::vector<std::string>& elems) {
  data["decls"]["type_decls"].push_back(detail::Union{detail::e_machines_t, elems});
}

void MurphiCodeGen::_typeMessage() {
  json msgJson = {
      {"id", detail::r_message_t},
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
    return detail::sr_cache_val_t;
  }
  if (t.isa<mlir::pcc::StateType>()) {
    return mach == detail::machines.cache ? e_cache_state_t()
                                          : e_directory_state_t();
  }
  if (t.isa<mlir::pcc::IDType>()) {
    return detail::e_machines_t;
  }
  // TODO - add support for more types
  assert(0 && "currently using an unsupported type!");
}
} // namespace murphi
