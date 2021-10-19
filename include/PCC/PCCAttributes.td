#ifndef PCC_ATTRIBUTES
#define PCC_ATTRIBUTES

include "mlir/IR/SubElementInterfaces.td"



def StateAttr : AttrDef<PCC_Dialect, "State", [], "::mlir::Attribute">
{
    let summary = "A attribute that holds a state";
    let description = [{}];
    let mnemonic = "state_attr";

    let parameters = (ins "std::string":$attrData, AttributeSelfTypeParameter<"StateType">:$type);
    let extraClassDeclaration = [{
    }];

    let builders = [
        AttrBuilderWithInferredContext<(ins "StateType":$theType), [{
            return $_get(theType.getContext(), theType.getState(), theType);
        }]>,
        AttrBuilder<(ins "std::string":$state),[{
            return get(StateType::get($_ctxt, state));
        }]>
    ];
}

#endif // PCC_ATTRIBUTES