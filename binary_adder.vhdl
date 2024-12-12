library IEEE;
use IEEE.std_logic_1164.all;

entity mtncl_netlist is
  port (
    input_0 : in std_logic;
    input_1 : in std_logic;
    input_2 : in std_logic;
    input_3 : in std_logic;
    output_0 : out std_logic;
    output_1 : out std_logic;
    output_2 : out std_logic;
    reset : in std_logic;  -- Active-low reset
    ki : in std_logic;     -- Input completion
    ko : out std_logic;    -- Output completion
    sleep : in std_logic   -- Sleep signal
  );
end entity mtncl_netlist;

architecture behavioral of mtncl_netlist is
  component TH23
    port (
      a : in std_logic_vector(19 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH34w3
    port (
      a : in std_logic_vector(19 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH33w2
    port (
      a : in std_logic_vector(5 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH23w2
    port (
      a : in std_logic_vector(5 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component THxor0
    port (
      a : in std_logic_vector(5 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH24w2
    port (
      a : in std_logic_vector(5 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component THand0
    port (
      a : in std_logic_vector(5 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH44w2
    port (
      a : in std_logic_vector(13 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH12
    port (
      a : in std_logic_vector(13 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH24
    port (
      a : in std_logic_vector(5 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  -- Internal signals
  signal gate_l1_7_out : std_logic;
  signal gate_l1_7_ko : std_logic;
  signal gate_l1_8_out : std_logic;
  signal gate_l1_8_ko : std_logic;
  signal gate_l1_9_out : std_logic;
  signal gate_l1_9_ko : std_logic;
  signal gate_l1_10_out : std_logic;
  signal gate_l1_10_ko : std_logic;
  signal gate_l1_11_out : std_logic;
  signal gate_l1_11_ko : std_logic;
  signal gate_l1_12_out : std_logic;
  signal gate_l1_12_ko : std_logic;
  signal gate_l1_13_out : std_logic;
  signal gate_l1_13_ko : std_logic;
  signal gate_l1_14_out : std_logic;
  signal gate_l1_14_ko : std_logic;
  signal gate_l2_15_out : std_logic;
  signal gate_l2_15_ko : std_logic;
  signal gate_l2_16_out : std_logic;
  signal gate_l2_16_ko : std_logic;
  signal gate_l2_17_out : std_logic;
  signal gate_l2_17_ko : std_logic;
  signal gate_l2_18_out : std_logic;
  signal gate_l2_18_ko : std_logic;
  signal gate_l2_19_out : std_logic;
  signal gate_l2_19_ko : std_logic;
  signal gate_l2_20_out : std_logic;
  signal gate_l2_20_ko : std_logic;
  signal output_0_21_out : std_logic;
  signal output_0_21_ko : std_logic;
  signal output_1_22_out : std_logic;
  signal output_1_22_ko : std_logic;
  signal output_2_23_out : std_logic;
  signal output_2_23_ko : std_logic;
  constant VDD : std_logic := '1';
  constant GND : std_logic := '0';
begin

  -- gate_l1_7 (TH23w2)
  gate_l1_7_inst: TH23w2
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_7_ko,
      z => gate_l1_7_out
    );

  -- gate_l1_8 (TH23w2)
  gate_l1_8_inst: TH23w2
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_8_ko,
      z => gate_l1_8_out
    );

  -- gate_l1_9 (TH24)
  gate_l1_9_inst: TH24
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_9_ko,
      z => gate_l1_9_out
    );

  -- gate_l1_10 (TH24w2)
  gate_l1_10_inst: TH24w2
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_10_ko,
      z => gate_l1_10_out
    );

  -- gate_l1_11 (TH33w2)
  gate_l1_11_inst: TH33w2
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_11_ko,
      z => gate_l1_11_out
    );

  -- gate_l1_12 (THxor0)
  gate_l1_12_inst: THxor0
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_12_ko,
      z => gate_l1_12_out
    );

  -- gate_l1_13 (TH23w2)
  gate_l1_13_inst: TH23w2
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_13_ko,
      z => gate_l1_13_out
    );

  -- gate_l1_14 (THand0)
  gate_l1_14_inst: THand0
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_15_ko,
      ko => gate_l1_14_ko,
      z => gate_l1_14_out
    );

  -- gate_l2_15 (TH24)
  gate_l2_15_inst: TH24
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_21_ko,
      ko => gate_l2_15_ko,
      z => gate_l2_15_out
    );

  -- gate_l2_16 (THxor0)
  gate_l2_16_inst: THxor0
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_21_ko,
      ko => gate_l2_16_ko,
      z => gate_l2_16_out
    );

  -- gate_l2_17 (THand0)
  gate_l2_17_inst: THand0
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_21_ko,
      ko => gate_l2_17_ko,
      z => gate_l2_17_out
    );

  -- gate_l2_18 (TH44w2)
  gate_l2_18_inst: TH44w2
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_21_ko,
      ko => gate_l2_18_ko,
      z => gate_l2_18_out
    );

  -- gate_l2_19 (THxor0)
  gate_l2_19_inst: THxor0
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_21_ko,
      ko => gate_l2_19_ko,
      z => gate_l2_19_out
    );

  -- gate_l2_20 (TH12)
  gate_l2_20_inst: TH12
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_21_ko,
      ko => gate_l2_20_ko,
      z => gate_l2_20_out
    );

  -- output_0_21 (THand0)
  output_0_21_inst: THand0
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out, gate_l2_15_out, gate_l2_16_out, gate_l2_17_out, gate_l2_18_out, gate_l2_19_out, gate_l2_20_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output_0
    );

  -- output_1_22 (TH34w3)
  output_1_22_inst: TH34w3
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out, gate_l2_15_out, gate_l2_16_out, gate_l2_17_out, gate_l2_18_out, gate_l2_19_out, gate_l2_20_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output_1
    );

  -- output_2_23 (TH23)
  output_2_23_inst: TH23
    port map (
      a => (input_0, input_1, input_2, input_3, VDD, GND, gate_l1_7_out, gate_l1_8_out, gate_l1_9_out, gate_l1_10_out, gate_l1_11_out, gate_l1_12_out, gate_l1_13_out, gate_l1_14_out, gate_l2_15_out, gate_l2_16_out, gate_l2_17_out, gate_l2_18_out, gate_l2_19_out, gate_l2_20_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output_2
    );
end architecture behavioral;
