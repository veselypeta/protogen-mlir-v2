#include "FSM/FSMDialect.h"

using namespace mlir;
using namespace fsm;

void FSMDialect::initialize() {
  addTypes<>();
  addOperations<>();
}