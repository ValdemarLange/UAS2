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
-- File                 : 01_button_led-vhd 
-- Date                 : 2026-01-12  
-- Revision             : 1.0  
-- Description          : Control a LED with a button to demonstrate that board works.  
-- Notes                : For students and project participants

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity Sindri  is                                     -- We call the entity Sindri .. as it more or less symbolize the Sindri PCB
    Port ( nBUTTON_I : in STD_LOGIC;                  -- The button draw signal LOW when pressed.
           LED_O : out STD_LOGIC_VECTOR(3 downto 0)); -- The LED lights up when output is HIGH           
end Sindri;

architecture Behavioral of Sindri is
begin
  LED_O(0)<=not nBUTTON_I; -- Led ON when button pressed
end Behavioral;
