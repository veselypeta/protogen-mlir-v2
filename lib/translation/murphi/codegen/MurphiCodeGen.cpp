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
 * Union to_json helper method
 */
void to_json(json &j, const Union &u) {
  j = {{"id", u.id}, {"typeId", "union"}, {"type", {{"listElems", u.elems}}}};
}

/*
 * Record to_json helper method
 */
void to_json(json &j, const Record &r) {
  json decls = json::array();
  std::for_each(std::begin(r.elems), std::end(r.elems), [&decls](auto &elem) {
    decls.push_back(
        {{"id", elem.first}, {"typeId", "ID"}, {"type", elem.second}});
  });
  j = {{"id", r.id}, {"typeId", "record"}, {"type", {{"decls", decls}}}};
}

/*
 * ScalarSet to_json helper functions
 */
void to_json(json &j, const ScalarSet<std::string> &ss) {
  j = {{"id", ss.id}, {"typeId", "scalarset"}, {"type", {{"type", ss.value}}}};
}
void to_json(json &j, const ScalarSet<size_t> &ss) {
  size_t val = ss.value;
  j = {{"id", ss.id}, {"typeId", "scalarset"}, {"type", {{"type", val}}}};
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
[[nodiscard]] static std::string e_directory_state_t() {
  return machines.directory.str() + state_suffix;
}
[[nodiscard]] static std::string e_cache_state_t() {
  return machines.cache.str() + state_suffix;
}
[[nodiscard]] static std::string r_cache_entry_t() {
  return std::string(EntryKey) + machines.cache.str();
}
[[nodiscard]] static std::string r_directory_entry_t() {
  return std::string(EntryKey) + machines.directory.str();
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
  _typeMachineSets();
  _typeMachineEntry();
  _typeMessage();
  _typeNetworkObjects();
}

mlir::LogicalResult MurphiCodeGen::render() {
  // validate json
  assert(detail::validateMurphiJSON(data) &&
         "JSON from codegen does not validate with the json schema");
  auto &env = InjaEnvSingleton::getInstance();
  auto tmpl = env.parse_template("murphi_base.tmpl");
  auto result = env.render(tmpl, data);
  output << result;
  return mlir::success();
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
      detail::e_cache_state_t(),
      moduleInterpreter.getEnumMachineStates(detail::machines.cache.str())});
  data["decls"]["type_decls"].push_back(detail::Enum{
      detail::e_directory_state_t(), moduleInterpreter.getEnumMachineStates(
                                         detail::machines.directory.str())});
}

void MurphiCodeGen::_typeStatics() {
  // Address type
  data["decls"]["type_decls"].push_back(detail::ScalarSet<std::string>{
      detail::ss_address_t, detail::c_val_cnt_t});
  // ClValue type
  // TODO - maybe create a sub-range struct
  data["decls"]["type_decls"].push_back(
      {{"id", detail::sr_cache_val_t},
       {"typeId", "sub_range"},
       {"type", {{"start", 0}, {"stop", detail::c_val_cnt_t}}}});
}

void MurphiCodeGen::_typeMachineSets() {
  auto add_mach_decl_type = [&](const std::string &machine,
                                const mlir::pcc::PCCType &type) {
    std::string typeIdent = machine == detail::machines.cache
                                ? detail::cache_set_t()
                                : detail::directory_set_t();
    if (type.isa<mlir::pcc::SetType>()) {
      auto setSize = type.cast<mlir::pcc::SetType>().getNumElements();
      data["decls"]["type_decls"].push_back(
          detail::ScalarSet<size_t>(detail::cache_set_t(), setSize));
    } else {
      // must be a struct type
      assert(type.isa<mlir::pcc::StructType>() &&
             "machine was not of set/struct type!");
      data["decls"]["type_decls"].push_back(detail::Enum{typeIdent, {machine}});
    }
  };

  // add the cache set
  add_mach_decl_type(detail::machines.cache.str(),
                     moduleInterpreter.getCache().getType());
  add_mach_decl_type(detail::machines.directory.str(),
                     moduleInterpreter.getDirectory().getType());

  // push a union of these for the Machines type
  data["decls"]["type_decls"].push_back(
      detail::Union{detail::e_machines_t,
                    {detail::cache_set_t(), detail::directory_set_t()}});
}

void MurphiCodeGen::_typeNetworkObjects() {
  for (const auto &type : detail::emitNetworkDefinitionJson()) {
    data["decls"]["type_decls"].push_back(type);
  }
}

void MurphiCodeGen::_typeMachineEntry() {
  // *** cache *** //
  mlir::pcc::CacheDeclOp cacheOp = moduleInterpreter.getCache();
  _getMachineEntry(cacheOp.getOperation());

  // *** directory *** //
  mlir::pcc::DirectoryDeclOp directoryOp = moduleInterpreter.getDirectory();
  _getMachineEntry(directoryOp.getOperation());
}

void MurphiCodeGen::_getMachineEntry(mlir::Operation *machineOp) {
  // Check that the operation is either cache or directory decl
  const auto opIdent = machineOp->getName().getIdentifier().strref();
  assert(opIdent == detail::opStringMap.cache_decl ||
         opIdent == detail::opStringMap.dir_decl &&
             "invalid operation passed to generation machine function");
  const auto machineIdent = opIdent == detail::opStringMap.cache_decl
                                ? detail::machines.cache
                                : detail::machines.directory;

  const auto is_id_attr = [](const mlir::NamedAttribute &attr) {
    return attr.first == "id";
  };

  std::vector<std::pair<std::string, std::string>> record_elems;

  const auto generate_mach_attr_field =
      [&record_elems, &machineIdent](const mlir::NamedAttribute &attr) {
        std::string fieldID = attr.first.str();
        mlir::TypeAttr typeAttr = attr.second.cast<mlir::TypeAttr>();
        std::string fieldType =
            MLIRTypeToMurphiTypeRef(typeAttr.getValue(), machineIdent);
        record_elems.emplace_back(fieldID, fieldType);
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

  std::string machine_id_t = machineIdent == detail::machines.cache
                                 ? detail::r_cache_entry_t()
                                 : detail::r_directory_entry_t();
  // generate the correct murphi declaration
  data["decls"]["type_decls"].push_back(
      detail::Record{machine_id_t, record_elems});
}

void MurphiCodeGen::_typeMessage() {

  auto msg_decl_type = get_glob_msg_type();

  // generate the glob msg
  detail::Record glob_msg = detail::Record{detail::r_message_t, msg_decl_type};

  data["decls"]["type_decls"].push_back(glob_msg);
}

std::vector<std::pair<std::string, std::string>> MurphiCodeGen::get_glob_msg_type(){
  std::map<std::string, std::string> glob_msg_type;
  auto all_msg_types = moduleInterpreter.getMessages();


  std::for_each(all_msg_types.begin(), all_msg_types.end(), [&](const mlir::pcc::MsgDeclOp &msgDeclOp){
    std::for_each(msgDeclOp->getAttrs().begin(), msgDeclOp->getAttrs().end(), [&](const mlir::NamedAttribute &named_attr){
      if(named_attr.first != "id"){ // skip the id field
        mlir::TypeAttr typeAttr = named_attr.second.cast<mlir::TypeAttr>();
        glob_msg_type.insert({named_attr.first.str(), MLIRTypeToMurphiTypeRef(typeAttr.getValue(), "")});
      }
    });
  });

  // copy to vector
  std::vector<std::pair<std::string, std::string>> msg_decl_type;
  msg_decl_type.reserve(glob_msg_type.size() + 1);
  for(auto &mv : glob_msg_type){
    msg_decl_type.emplace_back(mv);
  }

  // push back the address type also
  msg_decl_type.emplace_back(detail::c_adr, detail::ss_address_t );

  return msg_decl_type;
}

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t,
                                    const llvm::StringRef mach) {
  if (t.isa<mlir::pcc::DataType>()) {
    return detail::sr_cache_val_t;
  }
  if (t.isa<mlir::pcc::StateType>()) {
    return mach == detail::machines.cache ? detail::e_cache_state_t()
                                          : detail::e_directory_state_t();
  }
  if (t.isa<mlir::pcc::IDType>()) {
    return detail::e_machines_t;
  }
  if (t.isa<mlir::pcc::MsgIdType>()) {
    return detail::r_message_t;
  }
  // TODO - fix this implementation
  // we're better off if we declare a type for this first
  // and then referr to it.
  if (t.isa<mlir::pcc::SetType>()){
    // create a type that we can refer to. This may be more challenging now
    mlir::pcc::SetType st = t.cast<mlir::pcc::SetType>();

    return "multiset [ " + std::to_string(st.getNumElements()) + " ] of " + MLIRTypeToMurphiTypeRef(st.getElementType(), "");
  }

  if (t.isa<mlir::pcc::IntRangeType>()){
    mlir::pcc::IntRangeType intRangeType = t.cast<mlir::pcc::IntRangeType>();
    return std::to_string(intRangeType.getStartRange()) + ".." + std::to_string(intRangeType.getEndRange());
  }

  // TODO - add support for more types
  assert(0 && "currently using an unsupported type!");
}
} // namespace murphi
