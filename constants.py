FL4x4 = '4x4'
FL8x8 = '8x8'
FL20x20 = '20x20'

HANOI_3D = '3_Disks'

TERM_STATE_MAP = {
    FL4x4: [5, 7, 11, 12],
    FL8x8: [19, 29, 35, 41, 42, 46, 49, 52, 54, 59],
    FL20x20: [7,8,9,22,23,36,37,43,51,52,56,57,65,76,77,85,96,97,105,116,117,128,136,137,145,150,151,156,157,165,176,177,185,190,191,192,196,197,211,212,213,214,215,216,217,225,226,227,228,236,237,245,250,251,252,256,257,276,277,285,292,296,297,305,316,317,331,336,337,341,342,346,351,357,361,362,364,366,383,389,394,395,397]
}
GOAL_STATE_MAP = {
    FL4x4: [15],
    FL8x8: [63],
    FL20x20: [399]
}

#generated with c# code
#var s = "SFFFFFFHHHFFFFFFFFFFFFHHFFFFFFFFFFFFHHFFFFFHFFFFFFFHHFFFHHFFFFFFFHFFFFFFFFFFHHFFFFFFFHFFFFFFFFFFHHFFFFFFFHFFFFFFFFFFHHFFFFFFFFFFHFFFFFFFHHFFFFFFFHFFFFHHFFFFHHFFFFFFFHFFFFFFFFFFHHFFFFFFFHFFFFHHHFFFHHFFFFFFFFFFFFFHHHHHHHFFFFFFFHHHHFFFFFFFHHFFFFFFFHFFFFHHHFFFHHFFFFFFFFFFFFFFFFFFHHFFFFFFFHFFFFFFHFFFHHFFFFFFFHFFFFFFFFFFHHFFFFFFFFFFFFFHFFFFHHFFFHHFFFHFFFFHFFFFFHFFFHHFHFHFFFFFFFFFFFFFFFFHFFFFFHFFFFHHFHFG";
 
# for (int i = 0; i < s.Length; i++)
# {
#     if (s[i] == 'H')
#     {
#         Console.Write(i.ToString() + ",");
#     }
#}