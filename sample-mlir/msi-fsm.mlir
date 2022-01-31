%fwd = fsm.network {ordering="ordered", sym_name="fwd"} : !fsm.network
%resp = fsm.network {ordering="unordered", sym_name="resp"} : !fsm.network
%req = fsm.network {ordering="unordered", sym_name="req"} : !fsm.network

fsm.machine @cache(){
    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %acksReceived = fsm.variable "acksReceived" {initValue = 0} : !fsm.range<0, 3>
    %acksExpected = fsm.variable "acksExpected" {initValue = 0} : !fsm.range<0, 3>


}

fsm.machine @directory(){

    %State = fsm.variable "State" {initValue = "I"} : !fsm.state
    %cl = fsm.variable "cl" : !fsm.data
    %cache = fsm.variable "cache" : !fsm.set<!fsm.id, 3>
    %owner = fsm.variable "owner" : !fsm.id
}