library IEEE;
use IEEE.std_logic_1164.all;

entity mtncl_netlist is
  port (
    input_0 : in std_logic;
    input_1 : in std_logic;
    input_2 : in std_logic;
    input_3 : in std_logic;
    input_4 : in std_logic;
    input_5 : in std_logic;
    input_6 : in std_logic;
    input_7 : in std_logic;
    input_8 : in std_logic;
    output_0 : out std_logic;
    output_1 : out std_logic;
    output_2 : out std_logic;
    output_3 : out std_logic;
    reset : in std_logic;  -- Active-low reset
    ki : in std_logic;     -- Input completion
    ko : out std_logic;    -- Output completion
    sleep : in std_logic   -- Sleep signal
  );
end entity mtncl_netlist;

architecture behavioral of mtncl_netlist is
  component TH12
    port (
      a : in std_logic_vector(2 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component THand0
    port (
      a : in std_logic_vector(1 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component THxor0
    port (
      a : in std_logic_vector(1 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH33
    port (
      a : in std_logic_vector(2 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH24
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH23w2
    port (
      a : in std_logic_vector(2 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH34w2
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH22
    port (
      a : in std_logic_vector(1 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH34w3
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH24w2
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH23
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  component TH24w22
    port (
      a : in std_logic_vector(3 downto 0);
      sleep : in std_logic;
      rst : in std_logic;
      ki : in std_logic;
      ko : out std_logic;
      z : out std_logic
    );
  end component;

  -- Internal signals
  signal gate_l1_12_out : std_logic;
  signal gate_l1_12_ko : std_logic;
  signal gate_l1_13_out : std_logic;
  signal gate_l1_13_ko : std_logic;
  signal gate_l1_14_out : std_logic;
  signal gate_l1_14_ko : std_logic;
  signal gate_l1_16_out : std_logic;
  signal gate_l1_16_ko : std_logic;
  signal gate_l1_17_out : std_logic;
  signal gate_l1_17_ko : std_logic;
  signal gate_l1_18_out : std_logic;
  signal gate_l1_18_ko : std_logic;
  signal gate_l1_20_out : std_logic;
  signal gate_l1_20_ko : std_logic;
  signal gate_l1_21_out : std_logic;
  signal gate_l1_21_ko : std_logic;
  signal gate_l1_22_out : std_logic;
  signal gate_l1_22_ko : std_logic;
  signal gate_l1_23_out : std_logic;
  signal gate_l1_23_ko : std_logic;
  signal gate_l1_24_out : std_logic;
  signal gate_l1_24_ko : std_logic;
  signal gate_l1_26_out : std_logic;
  signal gate_l1_26_ko : std_logic;
  signal gate_l1_27_out : std_logic;
  signal gate_l1_27_ko : std_logic;
  signal gate_l2_28_out : std_logic;
  signal gate_l2_28_ko : std_logic;
  signal gate_l2_29_out : std_logic;
  signal gate_l2_29_ko : std_logic;
  signal gate_l2_31_out : std_logic;
  signal gate_l2_31_ko : std_logic;
  signal gate_l2_32_out : std_logic;
  signal gate_l2_32_ko : std_logic;
  signal gate_l2_33_out : std_logic;
  signal gate_l2_33_ko : std_logic;
  signal gate_l2_34_out : std_logic;
  signal gate_l2_34_ko : std_logic;
  signal gate_l2_36_out : std_logic;
  signal gate_l2_36_ko : std_logic;
  signal gate_l2_37_out : std_logic;
  signal gate_l2_37_ko : std_logic;
  signal gate_l2_38_out : std_logic;
  signal gate_l2_38_ko : std_logic;
  signal gate_l2_39_out : std_logic;
  signal gate_l2_39_ko : std_logic;
  signal gate_l3_41_out : std_logic;
  signal gate_l3_41_ko : std_logic;
  signal gate_l3_42_out : std_logic;
  signal gate_l3_42_ko : std_logic;
  signal gate_l3_43_out : std_logic;
  signal gate_l3_43_ko : std_logic;
  signal gate_l3_44_out : std_logic;
  signal gate_l3_44_ko : std_logic;
  signal gate_l3_45_out : std_logic;
  signal gate_l3_45_ko : std_logic;
  signal gate_l3_46_out : std_logic;
  signal gate_l3_46_ko : std_logic;
  signal gate_l3_47_out : std_logic;
  signal gate_l3_47_ko : std_logic;
  signal output_0_49_out : std_logic;
  signal output_0_49_ko : std_logic;
  signal output_1_50_out : std_logic;
  signal output_1_50_ko : std_logic;
  signal output_2_51_out : std_logic;
  signal output_2_51_ko : std_logic;
  constant VDD : std_logic := '1';
  constant GND : std_logic := '0';
begin

  -- gate_l1_12 (TH24w22)
  gate_l1_12_inst: TH24w22
    port map (
      a => (input_2, VDD),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_32_ko,
      ko => gate_l1_12_ko,
      z => gate_l1_12_out
    );

  -- gate_l1_13 (THxor0)
  gate_l1_13_inst: THxor0
    port map (
      a => (input_1),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_31_ko,
      ko => gate_l1_13_ko,
      z => gate_l1_13_out
    );

  -- gate_l1_14 (THand0)
  gate_l1_14_inst: THand0
    port map (
      a => (),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l1_14_ko,
      z => gate_l1_14_out
    );

  -- gate_l1_16 (TH23)
  gate_l1_16_inst: TH23
    port map (
      a => (input_2, VDD),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_46_ko,
      ko => gate_l1_16_ko,
      z => gate_l1_16_out
    );

  -- gate_l1_17 (TH33)
  gate_l1_17_inst: TH33
    port map (
      a => (input_2),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l1_17_ko,
      z => gate_l1_17_out
    );

  -- gate_l1_18 (TH24w22)
  gate_l1_18_inst: TH24w22
    port map (
      a => (VDD),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_29_ko,
      ko => gate_l1_18_ko,
      z => gate_l1_18_out
    );

  -- gate_l1_20 (TH22)
  gate_l1_20_inst: TH22
    port map (
      a => (GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_28_ko,
      ko => gate_l1_20_ko,
      z => gate_l1_20_out
    );

  -- gate_l1_21 (THand0)
  gate_l1_21_inst: THand0
    port map (
      a => (input_2),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_29_ko,
      ko => gate_l1_21_ko,
      z => gate_l1_21_out
    );

  -- gate_l1_22 (TH12)
  gate_l1_22_inst: TH12
    port map (
      a => (input_2, VDD),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l1_22_ko,
      z => gate_l1_22_out
    );

  -- gate_l1_23 (TH23)
  gate_l1_23_inst: TH23
    port map (
      a => (input_2, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_35_ko,
      ko => gate_l1_23_ko,
      z => gate_l1_23_out
    );

  -- gate_l1_24 (TH24w2)
  gate_l1_24_inst: TH24w2
    port map (
      a => (input_0, GND, VDD),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_40_ko,
      ko => gate_l1_24_ko,
      z => gate_l1_24_out
    );

  -- gate_l1_26 (TH12)
  gate_l1_26_inst: TH12
    port map (
      a => (input_3, VDD),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_30_ko,
      ko => gate_l1_26_ko,
      z => gate_l1_26_out
    );

  -- gate_l1_27 (THand0)
  gate_l1_27_inst: THand0
    port map (
      a => (input_1, GND),
      sleep => sleep,
      rst => reset,
      ki => gate_l2_32_ko,
      ko => gate_l1_27_ko,
      z => gate_l1_27_out
    );

  -- gate_l2_28 (TH33)
  gate_l2_28_inst: TH33
    port map (
      a => (gate_l1_20_out, VDD),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_43_ko,
      ko => gate_l2_28_ko,
      z => gate_l2_28_out
    );

  -- gate_l2_29 (TH22)
  gate_l2_29_inst: TH22
    port map (
      a => (gate_l1_18_out, gate_l1_21_out, VDD),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l2_29_ko,
      z => gate_l2_29_out
    );

  -- gate_l2_31 (TH23)
  gate_l2_31_inst: TH23
    port map (
      a => (gate_l1_18_out, input_2, gate_l1_13_out, gate_l1_21_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l2_31_ko,
      z => gate_l2_31_out
    );

  -- gate_l2_32 (THxor0)
  gate_l2_32_inst: THxor0
    port map (
      a => (gate_l1_12_out, VDD, gate_l1_27_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l2_32_ko,
      z => gate_l2_32_out
    );

  -- gate_l2_33 (TH24w2)
  gate_l2_33_inst: TH24w2
    port map (
      a => (GND, VDD, gate_l1_18_out, gate_l1_26_out),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_48_ko,
      ko => gate_l2_33_ko,
      z => gate_l2_33_out
    );

  -- gate_l2_34 (TH12)
  gate_l2_34_inst: TH12
    port map (
      a => (gate_l1_20_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l2_34_ko,
      z => gate_l2_34_out
    );

  -- gate_l2_36 (TH22)
  gate_l2_36_inst: TH22
    port map (
      a => (VDD, input_1, gate_l1_13_out),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_46_ko,
      ko => gate_l2_36_ko,
      z => gate_l2_36_out
    );

  -- gate_l2_37 (TH23w2)
  gate_l2_37_inst: TH23w2
    port map (
      a => (VDD, gate_l1_12_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l2_37_ko,
      z => gate_l2_37_out
    );

  -- gate_l2_38 (TH22)
  gate_l2_38_inst: TH22
    port map (
      a => (input_2, VDD, gate_l1_18_out),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_43_ko,
      ko => gate_l2_38_ko,
      z => gate_l2_38_out
    );

  -- gate_l2_39 (TH24)
  gate_l2_39_inst: TH24
    port map (
      a => (input_2, gate_l1_18_out),
      sleep => sleep,
      rst => reset,
      ki => gate_l3_44_ko,
      ko => gate_l2_39_ko,
      z => gate_l2_39_out
    );

  -- gate_l3_41 (TH23)
  gate_l3_41_inst: TH23
    port map (
      a => (GND, gate_l1_27_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_49_ko,
      ko => gate_l3_41_ko,
      z => gate_l3_41_out
    );

  -- gate_l3_42 (TH23w2)
  gate_l3_42_inst: TH23w2
    port map (
      a => (gate_l1_18_out, gate_l1_13_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l3_42_ko,
      z => gate_l3_42_out
    );

  -- gate_l3_43 (TH24w2)
  gate_l3_43_inst: TH24w2
    port map (
      a => (gate_l2_28_out, gate_l2_38_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l3_43_ko,
      z => gate_l3_43_out
    );

  -- gate_l3_44 (TH22)
  gate_l3_44_inst: TH22
    port map (
      a => (VDD, gate_l2_39_out),
      sleep => sleep,
      rst => reset,
      ki => output_0_49_ko,
      ko => gate_l3_44_ko,
      z => gate_l3_44_out
    );

  -- gate_l3_45 (TH34w2)
  gate_l3_45_inst: TH34w2
    port map (
      a => (gate_l1_13_out, VDD, gate_l1_23_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l3_45_ko,
      z => gate_l3_45_out
    );

  -- gate_l3_46 (THand0)
  gate_l3_46_inst: THand0
    port map (
      a => (gate_l2_36_out, gate_l1_16_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l3_46_ko,
      z => gate_l3_46_out
    );

  -- gate_l3_47 (TH34w3)
  gate_l3_47_inst: TH34w3
    port map (
      a => (input_1, gate_l1_21_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => gate_l3_47_ko,
      z => gate_l3_47_out
    );

  -- output_0_49 (TH22)
  output_0_49_inst: TH22
    port map (
      a => (gate_l3_44_out, gate_l3_41_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output_0
    );

  -- output_1_50 (TH22)
  output_1_50_inst: TH22
    port map (
      a => (gate_l3_44_out, gate_l1_21_out),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output_1
    );

  -- output_2_51 (THand0)
  output_2_51_inst: THand0
    port map (
      a => (),
      sleep => sleep,
      rst => reset,
      ki => ki,
      ko => ko,
      z => output_2
    );
end architecture behavioral;
