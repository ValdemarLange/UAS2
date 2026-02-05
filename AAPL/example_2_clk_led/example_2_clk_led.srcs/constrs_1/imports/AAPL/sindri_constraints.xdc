#    _             _                   __ _                                       _ 
#   /_\  _ __   __| | ___ _ __ ___    / _\ |_ ___ _ __   __ _  __ _  __ _ _ __ __| |
#  //_\\| '_ \ / _` |/ _ \ '__/ __|   \ \| __/ _ \ '_ \ / _` |/ _` |/ _` | '__/ _` |
# /  _  \ | | | (_| |  __/ |  \__ \   _\ \ ||  __/ | | | (_| | (_| | (_| | | | (_| |
# \_/ \_/_| |_|\__,_|\___|_|  |___/   \__/\__\___|_| |_|\__, |\__,_|\__,_|_|  \__,_|
#                                                       |___/                       
#  __                                                ____   ___ ____   __           
# / _\ ___/ _ __ ___ _ __  ___  ___ _ __            |___ \ / _ \___ \ / /_          
# \ \ / _ \| '__/ _ \ '_ \/ __|/ _ \ '_ \    _____    __) | | | |__) | '_ \         
# _\ \ (/) | | |  __/ | | \__ \  __/ | | |  |_____|  / __/| |_| / __/| (_) |        
# \__/\___/|_|  \___|_| |_|___/\___|_| |_|          |_____|\___/_____|\___/         
#     /                                             The "Sindri" FPGA project                                                 
#                                                   https://stengaard.net/Projects/Sindri/
#                               
# Sindri - common constraint file for Breadboard configuration
# Anders Stengaard SÃ¸rensen  January 2026
# V0.21 
# History: 
# 2026 01 10 V0.1  Created - on board pins defined
# 2026 01 22 V0.2  breadboard pins included - Vref property included
# 2026 01 28 V0.21 error corrected - BB_07 was rectified to pin J13 (was J11 by mistake)
 

# The following is important when a bit file is converted to a memory configuration file
# The property tell the "create memory configuration file" tool, that  a 4-bit wide SPI interface
# is used for the FLASH IC connected to the FPGA
# FLASH IC: IS25LP032D-JBLE  in 4-bit wide SPI configuration (see scematic)
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]

##### BOARD SPECIFIC PART
# Board: 'Sindri' Xilinx Spartan 7 experiment board - version 1.0 - January 2026 
#                                                     version 0.1 - December 2025 

# The button input                IOBANK34-IOB25 (IOB25 only has one pin)
set_property PACKAGE_PIN L5      [get_ports nBUTTON_I]
set_property IOSTANDARD LVCMOS33 [get_ports nBUTTON_I]

# -- The 4 LEDs                  IOBANK14
set_property PACKAGE_PIN P10     [get_ports {LED_O[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED_O[0]}]
set_property PACKAGE_PIN M11     [get_ports {LED_O[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED_O[1]}]
set_property PACKAGE_PIN N11     [get_ports {LED_O[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED_O[2]}]
set_property PACKAGE_PIN M12     [get_ports {LED_O[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED_O[3]}]

# The FTDI USB/UART pins          IOBANK14
set_property PACKAGE_PIN N10      [get_ports RX_I]   
set_property IOSTANDARD LVCMOS33  [get_ports RX_I]
set_property PACKAGE_PIN M10      [get_ports TX_O]
set_property IOSTANDARD LVCMOS33  [get_ports TX_O]


# 12MHz clock                    IOBANK14-IOB13-positive (MRCC-T2)
# --- The following define the 12MHz clk input pin/type/period/waveform and group
set_property PACKAGE_PIN H11     [get_ports CLK12_I]
set_property IOSTANDARD LVCMOS33 [get_ports CLK12_I]
create_clock -period 83.333 -name CLK12_I -waveform {0.000 41.166} [get_ports CLK12_I]
set_clock_groups -name XCLK -asynchronous -group [get_clocks CLK12_I]


#######################
# Breadboard connections
# Using the outer row of pads on J4
# Naming them _01 to _32 starting in the end closest to the USB connector
# Note: the signals could also be referred to as:
#  [get_ports {BB[5]}] ... [get_ports {BB[30]}]
#  which would allow referring to the entire set as a standlard logic vector BB(5 to 30) in VHDL
#  ... but that would mean that the entire set will have the same direction ... either INPUT, OUTPUT or INOUT

# Breadboard pincount 5 => J4 pad 10
set_property PACKAGE_PIN  P11     [get_ports BB_05]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_05]

# Breadboard pincount 6 => J4 pad 12
set_property PACKAGE_PIN  K11     [get_ports BB_06]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_06]

# Breadboard pincount 7 => J4 pad 14
set_property PACKAGE_PIN  J13     [get_ports BB_07]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_07]

# Breadboard pincount 8 => J4 pad 16
set_property PACKAGE_PIN  L13     [get_ports BB_08]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_08]

# Breadboard pincount 9 => J4 pad 18
set_property PACKAGE_PIN P13      [get_ports BB_09]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_09]

# Breadboard pincount 10 => J4 pad 20
set_property PACKAGE_PIN M13      [get_ports BB_10]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_10]

# Breadboard pincount 11 => J4 pad 22
set_property PACKAGE_PIN L12      [get_ports BB_11]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_11]

# Breadboard pincount 12 => J4 pad 24
set_property PACKAGE_PIN N14      [get_ports BB_12]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_12]

# Breadboard pincount 13 => J4 pad 26
set_property PACKAGE_PIN J11      [get_ports BB_13]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_13]

# Breadboard pincount 14 => J4 pad 28
set_property PACKAGE_PIN H14      [get_ports BB_14]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_14]

# Breadboard pincount 15 => J4 pad 30
set_property PACKAGE_PIN G14      [get_ports BB_15]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_15]

# Breadboard pincount 16 => J4 pad 32
set_property PACKAGE_PIN G11      [get_ports BB_16]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_16]

# Breadboard pincount 17 => J4 pad 34
set_property PACKAGE_PIN H13      [get_ports BB_17]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_17]

# Breadboard pincount 18 => J4 pad 36
set_property PACKAGE_PIN F13      [get_ports BB_18]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_18]

# Breadboard pincount 19 => J4 pad 38
set_property PACKAGE_PIN D14      [get_ports BB_19]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_19]

# Breadboard pincount 20 => J4 pad 40
set_property PACKAGE_PIN H12      [get_ports BB_20]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_20]

# Breadboard pincount 21 => J4 pad 42
set_property PACKAGE_PIN D13      [get_ports BB_21]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_21]

# Breadboard pincount 22 => J4 pad 44
set_property PACKAGE_PIN E12      [get_ports BB_22]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_22]

# Breadboard pincount 23 => J4 pad 46
set_property PACKAGE_PIN C12      [get_ports BB_23]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_23]

# Breadboard pincount 24 => J4 pad 48
set_property PACKAGE_PIN A13      [get_ports BB_24]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_24]

# Breadboard pincount 25 => J4 pad 50
set_property PACKAGE_PIN C4       [get_ports BB_25]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_25]

# Breadboard pincount 26 => J4 pad 52
set_property PACKAGE_PIN D4       [get_ports BB_26]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_26]

# Breadboard pincount 27 => J4 pad 54
set_property PACKAGE_PIN A5       [get_ports BB_27]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_27]

# Breadboard pincount 28 => J4 pad 56
set_property PACKAGE_PIN A4       [get_ports BB_28]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_28]

# Breadboard pincount 29 => J4 pad 58
set_property PACKAGE_PIN E4       [get_ports BB_29]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_29]

# Breadboard pincount 30 => J4 pad 60
set_property PACKAGE_PIN D3       [get_ports BB_30]   
set_property IOSTANDARD LVCMOS33  [get_ports BB_30]


#################################################
# Attributes that are only relevant for some applications
#

# The following is important when using differential Input
# It choose internal Vref, as the external Vref pins are allocated
# for normal I/O - and not designated as reference voltage input
# Choises are; 0.6 0.675 0.75 & 0.9
# (UG912 pp 256)
set_property INTERNAL_VREF 0.9 [get_iobanks 14] 
set_property INTERNAL_VREF 0.9 [get_iobanks 34] 







