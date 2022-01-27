#pragma once
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace murphi {
namespace detail {

/*
 * Some useful constants to have for compiling murphi
 */

// ** Naming Conventions **

// Suffixes
// _t : refers to a type
// _f : refers to a function name

// Prefixes
// c_   : refers to a constant
// e_   : refers to an enum
// ss_  : refers to a ScalarSet
// sr_  : refers to an integer SubRange
// r_   : refers to a record
// a_   : refers to an array

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
constexpr char CntKey[] = "cnt_";

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
constexpr char ss_cache_val_t[] = "ClValue";

// *** Alias Terms *** //
constexpr char adr_a[] = "adr";
constexpr char cle_a[] = "cle";

// *** Machine Keywords *** //
__attribute__((unused)) static std::string cache_set_t() {
  return std::string(SetKey) + machines.cache.str();
}

__attribute__((unused)) static std::string directory_set_t() {
  return std::string(SetKey) + machines.directory.str();
}

__attribute__((unused)) static std::string cache_mach_t() {
  return std::string(MachKey) + machines.cache.str();
}

__attribute__((unused)) static std::string directory_mach_t() {
  return std::string(MachKey) + machines.directory.str();
}

__attribute__((unused)) static std::string cache_obj_t() {
  return std::string(ObjKey) + machines.cache.str();
}

__attribute__((unused)) static std::string directory_obj_t() {
  return std::string(ObjKey) + machines.directory.str();
}

constexpr char e_machines_t[] = "Machines";

std::string e_directory_state_t();
std::string e_cache_state_t();
std::string r_cache_entry_t();
std::string r_directory_entry_t();

// *** Record Keywords *** //
constexpr char r_message_t[] = "Message";

// *** Built-in type *** //
constexpr char bool_t[] = "boolean";

// *** MSG ***
// default msg fields
constexpr auto c_adr = "adr";
constexpr auto c_mtype = "mtype";
constexpr auto c_src = "src";
constexpr auto c_dst = "dst";
constexpr auto c_msg = "msg";

constexpr auto c_state = "State";

constexpr char c_mach[] = "m";
constexpr char c_inmsg[] = "inmsg";

constexpr char a_cl_mutex_t[] = "CL_MUTEX";

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

// cpu events
constexpr auto cpu_events =
    std::array<llvm::StringRef, 3>{"load", "store", "evict"};

/*
 * VAR_DECL constants
 */
constexpr char mach_prefix_v[] = "i_";
std::string cache_v();
std::string directory_v();

constexpr char cl_mutex_v[] = "cl_mutex";

/*
 * Helper functions names
 */
constexpr char aq_mut_f[] = "Acquire_Mutex";
constexpr char rel_mut_f[] = "Release_Mutex";
constexpr char send_pref_f[] = "Send_";
constexpr auto cpu_action_pref_f = "SEND_";
constexpr char excess_messages_err[] = "Too many messages!";
constexpr char pop_pref_f[] = "Pop_";
constexpr char ordered_pop_err[] = "Trying to advance empty Q";

constexpr char mach_handl_pref_f[] = "Func_";

/*
 * Murphi Functions
 */
constexpr char multiset_add_f[] = "MultisetAdd";
constexpr auto multiset_remove_f = "MultiSetRemove";
constexpr auto is_member_f = "IsMember";
constexpr auto is_undefined_f = "isundefined";
} // namespace detail
} // namespace murphi