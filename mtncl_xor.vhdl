library IEEE;
use IEEE.std_logic_1164.all;

entity mtncl_xor is
  port (
    input_a, input_b : in std_logic;
    output : out std_logic;
    reset : in std_logic;  -- Active-low reset
    sleep : in std_logic   -- Sleep signal
  );
end entity mtncl_xor;

architecture behavioral of mtncl_xor is
  component TH34w2
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      z : out std_logic
    );
  end component;

  component TH24w22
    port (
      a : in std_logic_vector(7 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      z : out std_logic
    );
  end component;

  component THxor0
    port (
      a : in std_logic_vector(9 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      z : out std_logic
    );
  end component;

  component TH13
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      z : out std_logic
    );
  end component;

  component TH12
    port (
      a : in std_logic_vector(7 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      z : out std_logic
    );
  end component;

  component TH23w2
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      z : out std_logic
    );
  end component;

  -- Internal signals
  signal gate_l1_5_out : std_logic;
  signal gate_l1_5_ko : std_logic;
  signal gate_l1_6_out : std_logic;
  signal gate_l1_6_ko : std_logic;
  signal gate_l1_7_out : std_logic;
  signal gate_l1_7_ko : std_logic;
  signal gate_l1_8_out : std_logic;
  signal gate_l1_8_ko : std_logic;
  signal gate_l2_9_out : std_logic;
  signal gate_l2_9_ko : std_logic;
  signal gate_l2_10_out : std_logic;
  signal gate_l2_10_ko : std_logic;
  signal gate_l3_11_out : std_logic;
  signal gate_l3_11_ko : std_logic;
  constant VDD : std_logic := '1';
  constant GND : std_logic := '0';
begin

  -- gate_l1_5 (TH23w2)
  gate_l1_5_inst: TH23w2
    port map (
      a => (input_a, input_b, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_9_ko,
      ko => gate_l1_5_ko,
      z => gate_l1_5_out
    );

  -- gate_l1_6 (TH13)
  gate_l1_6_inst: TH13
    port map (
      a => (input_a, input_b, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_9_ko,
      ko => gate_l1_6_ko,
      z => gate_l1_6_out
    );

  -- gate_l1_7 (TH34w2)
  gate_l1_7_inst: TH34w2
    port map (
      a => (input_a, input_b, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_9_ko,
      ko => gate_l1_7_ko,
      z => gate_l1_7_out
    );

  -- gate_l1_8 (TH23w2)
  gate_l1_8_inst: TH23w2
    port map (
      a => (input_a, input_b, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_9_ko,
      ko => gate_l1_8_ko,
      z => gate_l1_8_out
    );

  -- gate_l2_9 (TH12)
  gate_l2_9_inst: TH12
    port map (
      a => (input_a, input_b, VDD, GND, gate_l1_5_out, gate_l1_6_out, gate_l1_7_out, gate_l1_8_out),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_11_ko,
      ko => gate_l2_9_ko,
      z => gate_l2_9_out
    );

  -- gate_l2_10 (TH24w22)
  gate_l2_10_inst: TH24w22
    port map (
      a => (input_a, input_b, VDD, GND, gate_l1_5_out, gate_l1_6_out, gate_l1_7_out, gate_l1_8_out),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_11_ko,
      ko => gate_l2_10_ko,
      z => gate_l2_10_out
    );

  -- gate_l3_11 (THxor0)
  gate_l3_11_inst: THxor0
    port map (
      a => (input_a, input_b, VDD, GND, gate_l1_5_out, gate_l1_6_out, gate_l1_7_out, gate_l1_8_out, gate_l2_9_out, gate_l2_10_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output
    );
end architecture behavioral;
