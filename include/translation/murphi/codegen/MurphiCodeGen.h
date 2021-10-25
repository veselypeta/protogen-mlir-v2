#pragma once
#include "inja/inja.hpp"
#include "mlir/IR/BuiltinDialect.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/utils/ModuleInterpreter.h"
#include <type_traits>

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

// a map to each type of machine
constexpr struct {
  const llvm::StringRef cache = "cache";
  const llvm::StringRef directory = "directory";
} machines;

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

// *** Machine Keywords *** //
static std::string cache_set_t() {
  return std::string(SetKey) + machines.cache.str();
}
static std::string directory_set_t() {
  return std::string(SetKey) + machines.directory.str();
}

static std::string cache_mach_t(){
  return std::string(MachKey) + machines.cache.str();
}

static std::string directory_mach_t(){
  return std::string(MachKey) + machines.directory.str();
}

static std::string cache_obj_t(){
  return std::string(ObjKey) + machines.cache.str();
}

static std::string directory_obj_t(){
  return std::string(ObjKey) + machines.directory.str();
}

constexpr char e_machines_t[] = "Machines";

static std::string e_directory_state_t();
[[nodiscard]] static std::string e_cache_state_t();
[[nodiscard]] static std::string r_cache_entry_t();
[[nodiscard]] static std::string r_directory_entry_t();

// *** Record Keywords *** //
constexpr char r_message_t[] = "Message";

// *** MSG ***
// default msg fields
constexpr auto c_adr = "adr";
constexpr auto c_mtype = "mtype";
constexpr auto c_src = "src";
constexpr auto c_dst = "dst";

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

/*
 * Helper Structs to generate JSON
 */

// *** ConstDecl
struct ConstDecl {
  std::string id;
  size_t value;
};

void to_json(json &j, const ConstDecl &c);

// *** TypeDecl
template <class TypeT>
struct TypeDecl{
  std::string id;
  TypeT type;
};

template <class TypeT>
void to_json(json &j, const TypeDecl<TypeT> &typeDecl) {
  j = typeDecl.type;
  j["id"] = typeDecl.id;
}

//// **** Type Expressions ////

// *** Enum
struct Enum {
  std::vector<std::string> elems;
};

void to_json(json &j, const Enum &c);

// *** Union
struct Union {
  std::vector<std::string> elems;
};

void to_json(json &j, const Union &c);

// *** Record
struct Record {
  std::vector<std::pair<std::string, std::string>> elems;
};

void to_json(json &j, const Record &record);

// *** ScalarSet
template <class T>
struct is_string : public std::is_same<std::string, typename std::decay_t<T>> {
};

template <class T> class ScalarSet {
public:
  ScalarSet() = delete;
  explicit ScalarSet(T value) {
    static_assert(std::is_integral<T>() || is_string<T>::value,
                  "ScalarSet value must be integral or string");

    this->value = value;
  }
  T value;
};

void to_json(json &j, const ScalarSet<std::string> &ss);
void to_json(json &j, const ScalarSet<size_t> &ss);

// *** ID
struct ID {
  std::string id;
};
void to_json(json &j, const ID &id);

// *** Array
template <class IndexT, class TypeT>
struct Array {
  IndexT index;
  TypeT type;
};

template <class IndexT, class TypeT>
void to_json(json &j, const Array<IndexT, TypeT> &arr) {
  j = {{"typeId", "array"},
       {"type", {{"index", arr.index}, {"type", arr.type}}}};
}


// *** SubRange
template <class StartT, class StopT>
struct SubRange {
  StartT start;
  StopT stop;
};

template <class StartT, class StopT>
void to_json(json &j, const SubRange<StartT, StopT> &sub_range) {
  j = {{"typeId", "sub_range"},
       {"type", {{"start", sub_range.start}, {"stop", sub_range.stop}}}};
}

// *** Multiset
template <class IndexT, class TypeT>
struct Multiset {
  IndexT index;
  TypeT type;
};

template <class IndexT, class TypeT>
void to_json(json &j, const Multiset<IndexT, TypeT> &m){
  j = {
      {"typeId", "multiset"},
      {"type", {
                   {"index", m.index},
                   {"type", m.type}
               }}
  };
}


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

  mlir::LogicalResult render();

  // helpers

private:
  // *** Type Generation functions *** /
  void _typeEnums(); // generate access/msgtype/states enums
  void _typeStatics(); // address space/ cl
  void _typeMachineSets(); // set cache/directory
  void _typeMessage(); // glob msg
  void _typeMachines();
  void _typeNetworkObjects();
  // additional helpers
  void _getMachineEntry(mlir::Operation *machineOp);
  void _getMachineMach();
  void _getMachineObjs();

  std::vector<std::pair<std::string, std::string>> get_glob_msg_type();

  ModuleInterpreter moduleInterpreter;
  mlir::raw_ostream &output;
  json data;
};

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t, llvm::StringRef mach);
} // namespace murphi