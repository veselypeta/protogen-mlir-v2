#pragma once
#include "inja/inja.hpp"
#include "mlir/IR/BuiltinDialect.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/utils/ModuleInterpreter.h"

// For a manual on the Murphi Programming Language :
// https://www.cs.ubc.ca/~ajh/courses/cpsc513/assign-token/User.Manual
namespace murphi {
using namespace inja;

namespace detail {

/*
 * Some useful constants to have for compiling murphi
 */

// ** Naming Conventions **

// Suffixes
// _t : refers to a type

// Prefixes
// c_   : refers to a constant
// e_   : refers to an enum
// ss_  : refers to a ScalarSet
// sr_  : refers to an integer SubRange
// r_   : refers to a record

constexpr char state_suffix[] = "_state";

// *** CONST *** //
constexpr char c_val_cnt_t[] = "VAL_COUNT";
constexpr size_t c_val_max = 1;
constexpr char c_adr_cnt_t[] = "ADR_COUNT";
constexpr size_t c_adr_cnt = 1;

// *** Keys *** //
constexpr char SetKey[] = "OBJSET_";
constexpr char EntryKey[] = "ENTRY_";
constexpr char MachKey[] = "MACH_";
constexpr char ObjKey[] = "OBJ_";
constexpr char Initval[] = "INITVAL";
constexpr char CLIdent_t[] = "CL";

// Network Parameters
constexpr char c_ordered_t[] = "O_NET_MAX";
constexpr char c_unordered_t[] = "U_NET_MAX";
constexpr char ordered[] = "Ordered";
constexpr char ordered_cnt[] = "Orderedcnt";
constexpr char unordered[] = "Unordered";

// *** Enum Keywords **** //
constexpr char e_access_t[] = "Access";
constexpr char e_message_type_t[] = "MessageType";
constexpr char ss_address_t[] = "Address";
constexpr char sr_cache_val_t[] = "ClValue";
constexpr char e_machines_t[] = "Machines";

// *** Record Keywords *** //
constexpr char r_message_t[] = "Message";

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
    std::make_pair(c_adr, ss_address_t), // target address
    {c_mtype, e_message_type_t},         // message type
    {c_src, e_machines_t},               // source
    {c_dst, e_machines_t}                // destination
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
  const llvm::StringRef dir_decl = "pcc.directory_decl";
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

struct Union {
  std::string id;
  std::vector<std::string> elems;
};

void to_json(json &j, const Union &c);

/*
 * Helper Generating Functions
 */
json emitNetworkDefinitionJson();

bool validateMurphiJSON(const json &j);
} // namespace detail
/*
 * Murphi Translation Class
 */
class MurphiCodeGen {
public:
  MurphiCodeGen(mlir::ModuleOp op, mlir::raw_ostream &output)
      : moduleInterpreter{ModuleInterpreter{op}}, output{output} {
    // initialize decls
    data["decls"] = json::object();
  }

  mlir::LogicalResult translate();

  void generateConstants();
  void generateTypes();
  // validate json
  bool is_json_valid();

  // Type functions
  void _typeEnums();
  void _typeStatics();
  void _typeMachines();
  void _typeMachinesUnion(const std::vector<std::string>& elems);
  void _typeMessage();
  void _typeNetworkObjects();
  std::string _getMachineEntry(mlir::Operation *machineOp);

  mlir::LogicalResult render();

private:
  ModuleInterpreter moduleInterpreter;
  mlir::raw_ostream &output;
  json data;
};

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t, llvm::StringRef mach);
} // namespace murphi