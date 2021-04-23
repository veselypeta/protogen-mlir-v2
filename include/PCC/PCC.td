#ifndef PCC_DIALECT
#define PCC_DIALECT

include "mlir/IR/OpBase.td"


def PCC_Dialect : Dialect {
    let name = "pcc";
    let summary = "a dialect for the PCC language";
    let description = [{
        TODO - add a description
    }];
    let cppNamespace = "::mlir::pcc";
}

class PCC_Op<string mnemonic, list<OpTrait> traits = []> : Op<PCC_Dialect, mnemonic, traits>;

include "PCCTypes.td"
include "PCCOps.td"

#endif // PCC_DIALECT