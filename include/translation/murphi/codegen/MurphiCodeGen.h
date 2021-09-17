#pragma once
#include "inja/inja.hpp"
#include "mlir/IR/BuiltinDialect.h"
#include "translation/utils/ModuleInterpreter.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"

namespace murphi {
using namespace inja;

namespace detail{

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

/*
 * Helper Structs to generate JSON
 */

struct ConstDecl {
  std::string id;
  size_t value;
};

void to_json(json &j, const ConstDecl &c);

struct Enum {
  std::string id;
  std::vector<std::string> elems;
};

void to_json(json &j, const Enum &c);

json emitNetworkDefinitionJson();

bool validateMurphiJSON(const json &j);
}
/*
 * Murphi Translation Class
 */
class MurphiCodeGen {
public:
  MurphiCodeGen(mlir::ModuleOp op, mlir::raw_ostream &output)
      : moduleInterpreter{ModuleInterpreter{op}}, output{output} {}

  mlir::LogicalResult translate();

  void generateConstants();
  void generateTypes();

  // Type functions
  void _typeEnums();
  void _typeStatics();
  void _typeMachines();
  void _typeMessage();
  void _typeNetworkObjects();
  void _getMachineEntry(mlir::Operation *machineOp);


  mlir::LogicalResult render();

private:
  ModuleInterpreter moduleInterpreter;
  mlir::raw_ostream &output;
  json data;
};


// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t,
                                    llvm::StringRef mach);
}