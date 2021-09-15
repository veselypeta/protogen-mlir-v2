#include "translation/murphi/TranslatePCCToMurphi.h"
#include "PCC/PCCOps.h"
#include "inja/inja.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <algorithm>
#include <iostream>
<<<<<<< HEAD
#include "translation/utils/ModuleInterpreter.h"
=======
>>>>>>> 9da33eb576af1b609bcce525d4415db2a936ca53

using namespace mlir::pcc;

// private namespace for the implementation
namespace {
using namespace inja;

/*
 * Some useful constants to have for compiling murphi
 */

constexpr char state_suffix[] = "_state";

// *** CONST *** //
constexpr char c_val_cnt_id[] = "VAL_COUNT";
constexpr size_t c_val_max = 1;
constexpr char c_adr_cnt_id[] = "ADR_COUNT";
constexpr size_t c_adr_cnt = 1;

// *** Keys *** //
constexpr char SetKey[] = "OBJSET_";
constexpr char EntryKey[] = "ENTRY_";
constexpr char MachKey[] = "MACH_";
constexpr char ObjKey[] = "OBJ_";
constexpr char Initval[] = "INITVAL";
constexpr char CLIdent[] = "CL";

// Network Parameters
constexpr char c_ordered[] = "O_NET_MAX";
constexpr char c_unordered[] = "U_NET_MAX";
constexpr char ordered[] = "Ordered";
constexpr char ordered_cnt[] = "Orderedcnt";
constexpr char k_o_network[] = "onet_";
constexpr char unordered[] = "Unordered";
constexpr char k_u_network[] = "unet_";

// *** Enum Keywords **** //
constexpr char k_access[] = "Access";
constexpr char k_message_type[] = "MessageType";
constexpr char k_address[] = "Address";
constexpr char k_cache_val[] = "ClValue";
constexpr char k_machines[] = "Machines";

// *** Record Keywords *** //
constexpr char r_message[] = "Message";

// *** MSG ***
// default msg fields
constexpr char c_adr[] = "adr";
constexpr char c_mtype[] = "mtype";
constexpr char c_src[] = "src";
constexpr char c_dst[] = "dst";

constexpr char c_mach[] = "m";
constexpr char c_cle[] = "cle";
constexpr char c_dle[] = "dle";
constexpr char c_inmsg[] = "inmsg";

const std::array<std::pair<std::string, std::string>, 4> BaseMsg{
    std::make_pair(c_adr, k_address), // target address
    {c_mtype, k_message_type},        // message type
    {c_src, k_machines},              // source
    {c_dst, k_machines}               // destination
};

const std::vector<std::pair<std::string, std::string>> SuperMsg{};

// config parameters
constexpr size_t c_fifo_max = 1;
constexpr bool enable_fifo = false;
constexpr size_t c_ordered_size = c_adr_cnt * 3 * 2 * 2;
constexpr size_t c_unordered_size = c_adr_cnt * 3 * 2 * 2;

// a map to each type of pcc operation
constexpr struct {
  const llvm::StringRef constant = "pcc.constant";
  const llvm::StringRef net_decl = "pcc.net_decl";
  const llvm::StringRef cache_decl = "pcc.cache_decl";
  const llvm::StringRef dir_decl = "pcc.dir_decl";
} opStringMap;

// a map to each type of machine
constexpr struct {
  const llvm::StringRef cache = "cache";
  const llvm::StringRef directory = "directory";
} machines;

<<<<<<< HEAD
=======
class ModuleInterpreter {
public:
  explicit ModuleInterpreter(mlir::ModuleOp op) : theModule{op} {}

  std::vector<mlir::pcc::ConstantOp> getConstants() {
    return getOperations<mlir::pcc::ConstantOp>();
  }

  template <class T> std::vector<T> getOperations() {
    std::vector<T> ops;
    for (auto &op : getModuleBody()) {
      if (mlir::isa<T>(op)) {
        ops.push_back(mlir::dyn_cast<T>(op));
      }
    }
    return ops;
  };

  std::vector<mlir::pcc::NetDeclOp> getNetworks() {
    return getOperations<mlir::pcc::NetDeclOp>();
  }

  mlir::pcc::CacheDeclOp getCache() {
    auto cacheOps = getOperations<mlir::pcc::CacheDeclOp>();
    assert(cacheOps.size() == 1 &&
           "found zero or more than one cache decl operations");
    return cacheOps.at(0);
  }

  mlir::pcc::DirectoryDeclOp getDirectory() {
    auto directoryOps = getOperations<mlir::pcc::DirectoryDeclOp>();
    assert(directoryOps.size() == 1 &&
           "found zero or more directory decl operations!");
    return directoryOps.at(0);
  }

  // FIXME - stub implementation
  std::vector<std::string> getEnumMessageTypes() {
    return {"Get_M", "Fwd_Get_M", "GetM_Ack_D"};
  }

  // FIXME - stub implementation
  std::vector<std::string> getEnumMachineStates(const std::string &mach) {
    std::vector<std::string> states = {"I", "M", "I_load", "M_evict"};
    std::for_each(states.begin(), states.end(),
                  [&mach](auto &state) { state = mach + "_" + state; });
    return states;
  }

private:
  mlir::ModuleOp theModule;
  mlir::Block &getModuleBody() {
    return theModule.getOperation()->getRegion(0).front();
  }
};
>>>>>>> 9da33eb576af1b609bcce525d4415db2a936ca53

/*
 * Helper Structs to generate JSON
 */

struct ConstDecl {
  std::string id;
  size_t value;
};

void to_json(json &j, const ConstDecl &c) {
  j = {{"id", c.id}, {"value", c.value}};
}

struct Enum {
  std::string id;
  std::vector<std::string> elems;
};

void to_json(json &j, const Enum &c) {
  j = {{"id", c.id}, {"typeId", "enum"}, {"type", {{"decls", c.elems}}}};
}

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

/*
 * Murphi Translation Class
 */
class MurphiTranslateImpl {
public:
  MurphiTranslateImpl(mlir::ModuleOp op, mlir::raw_ostream &output)
      : moduleInterpreter{ModuleInterpreter{op}}, output{output} {}

  mlir::LogicalResult translate() {
    generateConstants();
    generateTypes();
    return render();
  }

  void generateConstants() {
    // add the defined constants
    for (auto constantOp : moduleInterpreter.getConstants()) {
      data["decls"]["const_decls"].push_back(
          ConstDecl{constantOp.id().str(), constantOp.val()});
    }

    // boilerplate constants
    data["decls"]["const_decls"].push_back(ConstDecl{c_val_cnt_id, c_val_max});
    data["decls"]["const_decls"].push_back(ConstDecl{c_adr_cnt_id, c_adr_cnt});
    data["decls"]["const_decls"].push_back(
        ConstDecl{c_ordered, c_ordered_size});
    data["decls"]["const_-decls"].push_back(
        ConstDecl{c_unordered, c_unordered_size});
  }

  void generateTypes() {
    _typeEnums();
    _typeStatics();
    _typeMachines();
    // TODO - type machines
    _typeMessage();
    _typeNetworkObjects();
  }

  [[nodiscard]] static std::string k_directory_state_t() {
    return machines.directory.str() + state_suffix;
  }

  [[nodiscard]] static std::string k_cache_state_t() {
    return machines.cache.str() + state_suffix;
  }

  void _typeEnums() {
    // access enum
    data["decls"]["type_decls"].push_back(
        Enum{k_access, {"none", "load", "store"}});
    // messages enum
    data["decls"]["type_decls"].push_back(
        Enum{k_message_type, moduleInterpreter.getEnumMessageTypes()});
    // cache state
    data["decls"]["type_decls"].push_back(
        Enum{k_cache_state_t(),
             moduleInterpreter.getEnumMachineStates(machines.cache.str())});
    data["decls"]["type_decls"].push_back(
        Enum{k_directory_state_t(),
             moduleInterpreter.getEnumMachineStates(machines.directory.str())});
  }

  void _typeStatics() {
    // Address type
    data["decls"]["type_decls"].push_back({{"id", k_address},
                                           {"typeId", "scalarset"},
                                           {"type", {{"type", c_adr_cnt_id}}}});
    // ClValue type
    data["decls"]["type_decls"].push_back(
        {{"id", k_cache_val},
         {"typeId", "sub_range"},
         {"type", {{"start", 0}, {"stop", c_val_cnt_id}}}});
  }

  void _typeNetworkObjects() {
    for (const auto &type : emitNetworkDefinitionJson()) {
      data["decls"]["type_decls"].push_back(type);
    }
  }

  void _typeMachines() {
    // *** cache *** //
    mlir::pcc::CacheDeclOp cacheOp = moduleInterpreter.getCache();
    _getMachineEntry(cacheOp.getOperation());

    // *** directory *** //
//    mlir::pcc::DirectoryDeclOp directoryOp = moduleInterpreter.getDirectory();
  }

  void _getMachineEntry(mlir::Operation *machineOp) {
    // Check that the operation is either cache or directory decl
    auto opIdent = machineOp->getName().getIdentifier().strref();
    assert(opIdent == opStringMap.cache_decl ||
           opIdent == opStringMap.dir_decl &&
               "invalid operation passed to generation machine function");
    auto machineIdent =
        opIdent == opStringMap.cache_decl ? machines.cache : machines.directory;
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
        {{"id", std::string(EntryKey) + machines.cache.str()},
         {"typeId", "record"},
         {"type", r_cache}});
  }

  static std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t,
                                             const llvm::StringRef mach) {
    if (t.isa<mlir::pcc::DataType>()) {
      return k_cache_val;
    }
    if (t.isa<mlir::pcc::StateType>()) {
      return mach == machines.cache ? k_cache_state_t() : k_directory_state_t();
    }
    // TODO - add support for more types
    assert(0 && "currently using an unsupported type!");
  }

  void _typeMessage() {
    json msgJson = {
        {"id", r_message},
        {"typeId", "record"},
        {"type", {{"decls", json::array() /* initialize an empty array */}}}};
    // default types
    for (auto &defMsgType : BaseMsg) {
      msgJson["type"]["decls"].push_back({{"id", defMsgType.first},
                                          {"typeId", "ID"},
                                          {"type", defMsgType.second}});
    }
    // extra types
    for (auto &adiType : SuperMsg) {
      msgJson["type"]["decls"].push_back(
          {{"id", adiType.first}, {"typeId", "ID"}, {"type", adiType.second}});
    }
    data["decls"]["type_decls"].push_back(msgJson);
  }

  mlir::LogicalResult render() {
    auto &env = InjaEnvSingleton::getInstance();
    auto tmpl = env.parse_template("murphi_base.tmpl");
    auto result = env.render(tmpl, data);
    output << result;
    return mlir::success();
  }

private:
  ModuleInterpreter moduleInterpreter;
  mlir::raw_ostream &output;
  json data;
};

} // namespace

namespace mlir {
void registerToMurphiTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-murphi",
      [](mlir::ModuleOp op, mlir::raw_ostream &output) {
        MurphiTranslateImpl murphiTranslate(op, output);
        return murphiTranslate.translate();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<PCCDialect>();
        registry.insert<mlir::StandardOpsDialect>();
      });
}

} // namespace mlir