# Define Address for XL320
MY_DXL                      = 'XL320'     
ADDR_CW_ANGLE_LIMIT         = 6
ADDR_CCW_ANGLE_LIMIT        = 8
ADDR_CONTROL_MODE           = 11
ADDR_SHUTDOWN               = 18
ADDR_TORQUE_ENABLE          = 24
ADDR_LED                    = 25
ADDR_GOAL_POSITION          = 30
ADDR_GOAL_SPEED             = 32
ADDR_PRESENT_POSITION       = 37
ADDR_PRESENT_TEMPERATURE    = 46
ADDR_HARDWARE_ERROR_STATUS  = 50

# Define Actual Values
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the CW Angle Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 1023      # Refer to the CCW Angle Limit of product eManual
BAUDRATE                    = 1000000   # Default Baudrate of XL-320 is 1Mbps
PROTOCOL_VERSION            = 2.0

# Define Status in Decimal
TORQUE_ENABLE               = 1     # Value for enabling the torque
TORQUE_DISABLE              = 0     # Value for disabling the torque
WHEEL_MODE                  = 1
JOINT_MODE                  = 2

# Define Useful Variables
DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold

# LED color
LED_RED                     = 1
LED_GREEN                   = 2
LED_YELLOW                  = 3
LED_BLUE                    = 4
LED_PURPLE                  = 5
LED_CYAN                    = 6
LED_WHITE                   = 7

# Jetson PIN number
JETSON_CTL_PIN              = 12