---
--- Created by killa.
--- DateTime: 17-8-12 上午10:22
---


function table.slice(tbl, first, last, step)
    ---
    ---Get a slice of the whole table
    ---
    local sliced = {}

    for i = first or 1, last or #tbl, step or 1 do
        sliced[#sliced+1] = tbl[i]
    end

    return sliced
end
