import Jetson.GPIO as GPIO
from . import config
from dynamixel_sdk import *

class GPIOPacketHandler(Protocol2PacketHandler):
    def __init__(self):
        super(GPIOPacketHandler, self).__init__()
        
    def txRxPacket(self, port, txpacket):
        rxpacket = None
        error = 0

        GPIO.output(config.JETSON_CTL_PIN, GPIO.LOW)
        # tx packet
        result = self.txPacket(port, txpacket)
        time.sleep(0.0001)
        if result != COMM_SUCCESS:
            return rxpacket, result, error

        # (Instruction == BulkRead or SyncRead) == this function is not available.
        if txpacket[PKT_INSTRUCTION] == INST_BULK_READ or txpacket[PKT_INSTRUCTION] == INST_SYNC_READ:
            result = COMM_NOT_AVAILABLE

        # (ID == Broadcast ID) == no need to wait for status packet or not available.
        # (Instruction == action) == no need to wait for status packet
        if txpacket[PKT_ID] == BROADCAST_ID or txpacket[PKT_INSTRUCTION] == INST_ACTION:
            port.is_using = False
            return rxpacket, result, error

        # set packet timeout
        if txpacket[PKT_INSTRUCTION] == INST_READ:
            port.setPacketTimeout(DXL_MAKEWORD(txpacket[PKT_PARAMETER0 + 2], txpacket[PKT_PARAMETER0 + 3]) + 11)
        else:
            port.setPacketTimeout(11)
            # HEADER0 HEADER1 HEADER2 RESERVED ID LENGTH_L LENGTH_H INST ERROR CRC16_L CRC16_H

        GPIO.output(config.JETSON_CTL_PIN, GPIO.HIGH)
        # rx packet
        while True:
            rxpacket, result = self.rxPacket(port)
            time.sleep(0.0001)
            if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
                break

        if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
            error = rxpacket[PKT_ERROR]

        return rxpacket, result, error