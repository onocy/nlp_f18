const express = require('express')
const path = require('path');
const fs = require('fs');
const app = express()

app.get('/', (req, res) => (
    res.sendFile(path.join(__dirname + '/index.html'))
));

app.get('/embeddings', (req, res) => (
    fs.readFile('output.txt', 'utf8', function(err, data) {  
        if (err) throw err;
        res.send(JSON.parse(data));
    })
));

app.listen(3000, () => (
    console.log(`Listening on port 3000!`)
));