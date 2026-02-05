--    _             _                   __ _                                       _ 
--   /_\  _ __   __| | ___ _ __ ___    / _\ |_ ___ _ __   __ _  __ _  __ _ _ __ __| |
--  //_\\| '_ \ / _` |/ _ \ '__/ __|   \ \| __/ _ \ '_ \ / _` |/ _` |/ _` | '__/ _` |
-- /  _  \ | | | (_| |  __/ |  \__ \   _\ \ ||  __/ | | | (_| | (_| | (_| | | | (_| |
-- \_/ \_/_| |_|\__,_|\___|_|  |___/   \__/\__\___|_| |_|\__, |\__,_|\__,_|_|  \__,_|
--                                                       |___/                       
--  __                                                ____   ___ ____   __           
-- / _\ ___/ _ __ ___ _ __  ___  ___ _ __            |___ \ / _ \___ \ / /_          
-- \ \ / _ \| '__/ _ \ '_ \/ __|/ _ \ '_ \    _____    __) | | | |__) | '_ \         
-- _\ \ (/) | | |  __/ | | \__ \  __/ | | |  |_____|  / __/| |_| / __/| (_) |        
-- \__/\___/|_|  \___|_| |_|___/\___|_| |_|          |_____|\___/_____|\___/         
--     /                                             The "Sindri" FPGA project                                                 
--                                                   https://stengaard.net/Projects/Sindri/
--                               
-- Company/Organisation : University of Southern Denmark  
-- Department           : The UAS Center
-- Author               : Anders Stengaard SÃ¸rensen
-- Project              : Xilinx Spartan-7 experiment Board  
-- Module               : LED/Button test
-- File                 : 02_clk_led.vhd 
-- Date                 : 2026-01-12  
-- Revision             : 1.0  
-- Description          : Control a LED with a buttion to demonstrate that board works.  
-- Notes                : For students and project participants

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;            -- Allow us to use signed and unsigned types etc (see below)

entity Sindri is                     -- We call the entity Sindri, as it more or less represent the Sindri PCB  
    Port ( CLK12_I   : in  STD_LOGIC;                         -- 12 MHz clock from crystal oscillator Y1
           nBUTTON_I : in  STD_LOGIC;                         -- The button draw signal LOW when pressed.
           LED_O     : out STD_LOGIC_VECTOR(3 downto 0));     -- The 4 LED's lights up when output is HIGH
end Sindri;

architecture Behavioral of Sindri is

alias clk is CLK12_I;

signal syscnt : unsigned(31 downto 0) := (others=>'0');      -- This type is from IEEE.NUMERIC_STD

begin

process(clk)
begin
  if clk'event and clk='1' then
    syscnt<=syscnt+1;
  end if;
end process;

-- Map selected bits from syscnt to the 4 LED's (and let the button play a role)
--
--         ( A(n)     and    B     ) or  C     
----------------------------------------------------------
LED_O(0)<=(syscnt(22) and syscnt(11)) or (not  nBUTTON_I); -- MODIFIED to count slower and be dimmer
LED_O(1)<=(syscnt(23) and syscnt(11)) or (not  nBUTTON_I);
LED_O(2)<=(syscnt(24) and syscnt(11)) or (not  nBUTTON_I);
LED_O(3)<=(syscnt(25) and syscnt(11)) or (not  nBUTTON_I);
--
-- If you press the button nBUTTON_I is '0'
--   .. so not nBUTTOB_I is '1'
--   .. so the or (not nBUTTON) make the expression '1' and hence the LED is at full light
-- Else - if you don't press the button
--   .. the bit you want to show is and'ed with bit(10)
--   .. so the LED light up if both are '1' 
-- The frequency of syscnt(10) is 12MHz/2**(10+1) is ca 6 kHz - You can NOT se it blink . it will just appear at 50% light intensity
--                  syscnt(19) is ca 12 Hz
--                  syscnt(20) is ca 6 Hz
--                  syscnt(21) is ca 3 Hz
--                  syscnt(22) is ca 1.5Hz
-- So when visibly blinking signals are ANDed with 6KHz (50% duty cycle) - they will still apear to blink vusibly - just only light up at 50%
            

end Behavioral;
