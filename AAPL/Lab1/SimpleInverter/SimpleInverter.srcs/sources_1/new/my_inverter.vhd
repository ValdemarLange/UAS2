library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity my_inverter  is                                     
    Port ( nBUTTON_I : in STD_LOGIC;                  -- The button draw signal LOW when pressed.
           BB_06 : out STD_LOGIC); -- Output on pin 6 (from usb side) - K11 - BB_06 for breadboard connection 06           
end my_inverter;

architecture Behavioral of my_inverter is
begin
  BB_06<=not nBUTTON_I; -- Led ON when button pressed
  -- IO pin 6 -> COM (fælles resistor) -> andet r ben -> LED -> GND (bb_02)
end Behavioral;