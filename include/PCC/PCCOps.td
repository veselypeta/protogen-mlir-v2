#ifndef PCC_OPS
#define PCC_OPS

def FooOp : PCC_Op<"foo"> {
    let summary = "foo op";
    let description = [{}];
    let results = (outs PCCType:$results);
    let assemblyFormat = [{
        attr-dict `:` type($results)
    }];
}

// PCC Op to declare a interconnect network (either ordered or unordered)
def NetDeclOp : PCC_Op<"net_decl">{
    let summary = "Declare a global (ordered or unordered) network used for the protocol";
    let description = [{
        Declare a global `net_decl` operation. Returns a NetworkType SSA value which can be used in
        other operations.
    }];
    let arguments = (ins StrAttr:$netId);
    let results = (outs NetworkType:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

#endif // PCC_OPS