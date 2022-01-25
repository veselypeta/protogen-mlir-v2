#include "translation/murphi/codegen/MurphiConstants.h"

namespace murphi{
namespace detail{

std::string e_directory_state_t() {
  return machines.directory.str() + state_suffix;
}
std::string e_cache_state_t() { return machines.cache.str() + state_suffix; }
std::string r_cache_entry_t() {
  return std::string(EntryKey) + machines.cache.str();
}
std::string r_directory_entry_t() {
  return std::string(EntryKey) + machines.directory.str();
}

std::string cache_v() { return mach_prefix_v + machines.cache.str(); }
std::string directory_v() { return mach_prefix_v + machines.directory.str(); }

}
}