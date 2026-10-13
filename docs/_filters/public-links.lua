-- Use published destinations for relative links in the standalone manuscript PDF.
local site = "https://turn1a.github.io/safety-classifier-transfer-risk/"

function Link(link)
  if not FORMAT:match("typst") then
    return nil
  end
  if link.target:match("^[%w+.-]+:") or link.target:match("^#") then
    return nil
  end
  local target = link.target:gsub("^%./", "")
  target = target:gsub("%.qmd([#?])", ".html%1"):gsub("%.qmd$", ".html")
  link.target = site .. target
  return link
end
