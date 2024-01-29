function VideoCard() {
    // URL of the Flask server streaming the video
    const videoStreamUrl = "http://localhost:5000/video_feed";
  
    return (
      <div style={{ maxWidth: '640px', margin: 'auto', padding: '20px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)' }}>
        <h2 style={{ textAlign: 'left', marginBottom: '20px' , fontWeight:"bold"}}>Live Feed</h2>
        <div style={{ position: 'relative', paddingBottom: '56.25%', height: '0' }}>
          <img
            src={videoStreamUrl}
            style={{ position: 'absolute', top: '0', left: '0', width: '100%', height: '100%', border: 'none' }}
            frameBorder="0"
            allowFullScreen
          ></img>
        </div>
      </div>
    );
  }
  
  export default VideoCard;