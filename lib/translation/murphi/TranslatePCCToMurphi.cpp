#include "translation/murphi/TranslatePCCToMurphi.h"
#include "PCC/PCCOps.h"
#include "inja/inja.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"

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

// config parameters
constexpr size_t c_fifo_max = 1;
constexpr bool enable_fifo = false;
constexpr size_t c_ordered_size = c_adr_cnt * 3 * 2 * 2;
constexpr size_t c_unordered_size = c_adr_cnt * 3 * 2 * 2;

constexpr struct {
  const llvm::StringRef constant = "pcc.constant";
  const llvm::StringRef net_decl = "pcc.net_decl";
} opStringMap;

class ModuleInterpreter {
public:
  explicit ModuleInterpreter(mlir::ModuleOp op) : theModule{op} {}

  std::vector<mlir::pcc::ConstantOp> getConstants() {
    std::vector<mlir::pcc::ConstantOp> constants;
    for (auto &op : getModuleBody()) {
      if (op.getName().getIdentifier().strref() == opStringMap.constant) {
        auto constantOp = mlir::dyn_cast<mlir::pcc::ConstantOp>(op);
        constants.push_back(constantOp);
      }
    }
    return constants;
  }

  std::vector<mlir::pcc::NetDeclOp> getNetworks() {
    std::vector<mlir::pcc::NetDeclOp> networks;
    for (auto &op : getModuleBody()) {
      if (op.getName().getIdentifier().strref() == opStringMap.net_decl) {
        auto netOp = mlir::dyn_cast<mlir::pcc::NetDeclOp>(op);
        networks.push_back(netOp);
      }
    }
    return networks;
  }

  //FIXME - stub implementation
  std::vector<std::string> getEnumMessageTypes(){
    return {"Get_M", "Fwd_Get_M", "GetM_Ack_D"};
  }

  // FIXME - stub implementation
  std::vector<std::string> getEnumMachineStates(std::string &mach){
    std::vector<std::string> states = {"I", "M", "I_load", "M_evict"};
    std::for_each(states.begin(), states.end(), [&mach](auto &state){
      state = mach + "_" + state;
    });
    return states;
  }

  // FIXME - stub implementation
  std::vector<std::string> getMachineKeys(){
    return {"cache", "directory"};
  }

private:
  mlir::ModuleOp theModule;
  mlir::Block &getModuleBody() {
    return theModule.getOperation()->getRegion(0).front();
  }
};

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
  j = {
      {"id", c.id},
      {"typeId", "enum"},
      {"type", {
                   {"decls", c.elems}
               }}
      };
}

json emitNetworkDefinitionJson(){
  auto ord_type_name = std::string(ObjKey) + ordered;
  auto ord_type_count_name = std::string(ObjKey) + ordered_cnt;
  auto un_ord_type_name = std::string(ObjKey) + unordered;
  json j = {
        {
            {"id", ord_type_name},
            {"typeId", "array"},
            {"type", {
                         {"index", {
                                        {"typeId", "ID"},
                                        {"type", k_machines}
                                   }},
                         {"type", {
                                    {"typeId", "array"},
                                    {"type", {
                                                 {"index", {
                                                              {"typeId", "sub_range"},
                                                              {"type", {
                                                                           {"start", 0},
                                                                           {"stop", std::string(c_ordered) + "-1"}
                                                                       }}
                                                          }},
                                                 {"type", {
                                                             {"typeId", "ID"},
                                                             {"type", r_message}
                                                           }}
                                             }
                                    }


                                   }}
                     }}
      },
      {
          {"id", ord_type_count_name},
          {"typeId", "array"},
          {"type", {
                       {"index", {
                                     {"typeId", "ID"},
                                     {"type", k_machines}
                                 }},
                       {"type", {
                                    {"typeId", "sub_range"},
                                    {"type", {
                                                 {"start", 0},
                                                 {"stop", c_unordered}
                                             }}

                                }}
                   }}
      },
      {
          {"id", un_ord_type_name},
          {"typeId", "array"},
          {"type", {
                       {"index", {
                                     {"typeId", "ID"},
                                     {"type", k_machines}
                                 }
                        },
                       {"type", {
                                    {"typeId", "multiset"},
                                    {"type", {
                                                 {"index", {
                                                               {"typeId", "ID"},
                                                               {"type", c_ordered}
                                                           }},
                                                 {"type", {
                                                              {"typeId", "ID"},
                                                              {"type", r_message}
                                                          }}
                                             }}
                                }}
                   }}
      }
  };
  return j;
}

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
    generateEnums();
    generateNetworkObjects();
  }

  void generateEnums() {
    // access enum
    data["decls"]["type_decls"].push_back(
        Enum{k_access, {"none", "load", "store"}});
    // messages enum
    data["decls"]["type_decls"].push_back(
        Enum{k_message_type, moduleInterpreter.getEnumMessageTypes()});
    // machines states
    for(auto &mach : moduleInterpreter.getMachineKeys()){
      data["decls"]["type_decls"].push_back(
          Enum{mach + state_suffix, moduleInterpreter.getEnumMachineStates(mach)}
          );
    }
  }

  void generateNetworkObjects() {
    for(const auto &type : emitNetworkDefinitionJson()){
      data["decls"]["type_decls"].push_back(type);
    }
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