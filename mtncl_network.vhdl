
library IEEE;
use IEEE.std_logic_1164.all;

entity MTNCLNeuralNetwork is
    port (
        a, b: in std_logic;
        result: out std_logic;
        reset, sleep: in std_logic
    );
end MTNCLNeuralNetwork;

architecture behavioral of MTNCLNeuralNetwork is

    component TH12
        port (
            a1, a2: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH13
        port (
            a1, a2, a3: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH14
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH22
        port (
            a1, a2: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH23
        port (
            a1, a2, a3: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH23w2
        port (
            a1, a2, a3: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH24
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH24w2
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH24w22
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH33
        port (
            a1, a2, a3: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH33w2
        port (
            a1, a2, a3: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH34
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH34w2
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH34w3
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH34w22
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH44
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH44w2
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH44w3
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component TH44w22
        port (
            a1, a2, a3, a4: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component THxor
        port (
            a1, a2: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component THand
        port (
            a1, a2: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    component THor
        port (
            a1, a2: in std_logic;
            z: out std_logic;
            sleep: in std_logic
        );
    end component;

    -- Internal signals
    signal layer1_out1: std_logic;
    signal layer1_out2: std_logic;
    signal layer1_out3: std_logic;
    signal layer2_out1: std_logic;
    signal layer2_out2: std_logic;

begin

    -- Layer 1

    gate_l1_1: TH12
        port map (
            a1 => a, a2 => b,
            z => layer1_out1,
            sleep => sleep
        );

    gate_l1_2: TH22
        port map (
            a1 => a, a2 => b,
            z => layer1_out2,
            sleep => sleep
        );

    gate_l1_3: THand
        port map (
            a1 => a, a2 => b,
            z => layer1_out3,
            sleep => sleep
        );

    -- Layer 2

    gate_l2_1: TH13
        port map (
            a1 => layer1_out1, a2 => layer1_out2, a3 => layer1_out3,
            z => layer2_out1,
            sleep => sleep
        );

    gate_l2_2: TH23
        port map (
            a1 => layer1_out1, a2 => layer1_out2, a3 => layer1_out3,
            z => layer2_out2,
            sleep => sleep
        );

    result <= layer2_out2;  -- Final output
end behavioral;
