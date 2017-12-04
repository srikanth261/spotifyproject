/**
 * Created by lware on 5/7/17.
 */

// Note: code was adapted from http://bl.ocks.org/ilyabo/2209220


Playlist = function(_data){
    console.log(_data);
    this.data = _data;
    this.displaydata = _data;
}

Playlist.prototype.wrangleData = function(selectSong, selectBucket){
    var vis = this;
    vis.selectedSong = selectSong;
    vis.selectedBucket = selectBucket;
    
    $( ".play" ).empty();
    vis.displaydata.forEach(function(d) {
        // console.log([flow.Origin, flow.Dest]);
        if (d.seed_id==vis.selectedSong & d.bucket_name==vis.selectedBucket){

            var outputFrame = '<tr><td class="song playlist-col input-song"><iframe src="https://open.spotify.com/embed?uri=spotify:track:' + d.track_ids + '&theme=black&view=list" width="100%" height="80" frameborder="0" allowtransparency="true"></iframe></td><td class="popularity-col">' + d.popularity + '</td></tr>'
            $( ".play" ).append(outputFrame);
        }
    });
}

