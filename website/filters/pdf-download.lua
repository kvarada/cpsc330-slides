-- A lecture page explicitly opts in; otherwise remove its PDF download link.
function Pandoc(doc)
  if doc.meta['publish-pdf'] == true then
    return doc
  end
  return doc:walk({
    Link = function(link)
      if link.classes:includes('pdf-download') then
        return {}
      end
    end
  })
end
