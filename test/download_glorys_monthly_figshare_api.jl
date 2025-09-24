using Downloads, JSON3, HTTP

figshare_id = 30193228

function get_figshare_download_url(article_id)
    api_url = "https://api.figshare.com/v2/articles/$article_id"
    
    try
        response = HTTP.get(api_url)
        if response.status == 200
            data = JSON3.read(String(response.body))
            return data.files
        else
            @warn "API request failed with status: $(response.status)"
            return nothing
        end
    catch e
        @warn "Error accessing Figshare API: $e"
        return nothing
    end
end

"""
    download_from_figshare_api(article_id::String, download_dir::String=".")

Downloads files from a Figshare article using the official API method.
"""
function download_from_figshare_api(article_id, download_dir::String=".")
    files_info = get_figshare_download_url(article_id)
    
    if files_info === nothing
        @error "Could not get file information for article $article_id"
        return false
    end
    
    if !isdir(download_dir)
        mkpath(download_dir)
    end
    
    success_count = 0
    
    for file in files_info
        download_url = file.download_url
        filename = file.name
        filepath = joinpath(download_dir, filename)
        
        println("Downloading: $filename ($(file.size) bytes)")
        
        # Use proper headers to avoid being blocked
        headers = [
            "User-Agent" => "Julia Scientific Computing Client/1.0",
            "Accept" => "*/*",
            "Referer" => "https://figshare.com"
        ]
        
        try
            Downloads.download(download_url, filepath; headers=headers, timeout=120)
            println("✓ Successfully downloaded: $filename")
            success_count += 1
        catch e
            @warn "Failed to download $filename: $e"
        end
    end
    
    println("Downloaded $success_count/$(length(files_info)) files")
    return success_count > 0
end

download_from_figshare_api(figshare_id, "./GLORYS_data")