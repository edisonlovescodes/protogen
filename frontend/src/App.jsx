import React, { useState } from 'react';

function App() {
  const [generatedSequences, setGeneratedSequences] = useState([]);
  const [loading, setLoading] = useState(false);
  const [structureLoading, setStructureLoading] = useState(false);
  const [selectedSequence, setSelectedSequence] = useState(null);
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(1.0);
  const [numSequences, setNumSequences] = useState(1);
  const [properties, setProperties] = useState({
    thermostable: 0.5,
    enzyme: 0.5,
    membrane: 0.5,
    structural: 0.5,
    flexible: 0.5
  });

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          max_length: maxLength, 
          temperature: temperature, 
          num_sequences: numSequences,
          properties: properties
        })
      });
      const data = await res.json();
      setGeneratedSequences(data.sequences);
    } catch (e) {
      alert('Error: ' + e.message);
    }
    setLoading(false);
  };

  const handlePredictStructure = async (sequence, index) => {
    setStructureLoading(true);
    setSelectedSequence(index);
    
    try {
      const res = await fetch('http://localhost:8000/predict_structure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: sequence })
      });
      const data = await res.json();
      
      if (data.success) {
        alert('✅ 3D structure predicted! Opening viewer...');
        const encodedPdb = encodeURIComponent(data.pdb);
        window.open(`https://molstar.org/viewer/?pdb-data=${encodedPdb}`, '_blank');
      } else {
        alert('Error: ' + data.error);
      }
    } catch (e) {
      alert('Error: ' + e.message);
    }
    
    setStructureLoading(false);
  };

  const handleExport = async () => {
    if (generatedSequences.length === 0) {
      alert('Generate some sequences first!');
      return;
    }

    try {
      const res = await fetch('http://localhost:8000/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequences: generatedSequences })
      });
      const data = await res.json();
      
      const blob = new Blob([data.fasta], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'protogen_sequences.fasta';
      a.click();
    } catch (e) {
      alert('Export error: ' + e.message);
    }
  };

  const PropertySlider = ({ name, label, value, onChange }) => (
    <div style={{ marginBottom: '20px' }}>
      <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#555', fontSize: '14px' }}>
        {label}: {value.toFixed(2)}
      </label>
      <input 
        type="range" 
        min="0" 
        max="1" 
        step="0.05"
        value={value}
        onChange={(e) => onChange(name, parseFloat(e.target.value))}
        style={{ width: '100%', height: '6px', borderRadius: '5px', background: '#ddd', cursor: 'pointer', accentColor: '#667eea' }}
      />
    </div>
  );

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <div style={{ background: 'rgba(0,0,0,0.2)', padding: '60px 20px', textAlign: 'center', color: 'white' }}>
        <h1 style={{ fontSize: '48px', marginBottom: '10px', fontWeight: '700' }}>🧬 Protogen</h1>
        <p style={{ fontSize: '18px', opacity: 0.9 }}>AI-Powered Protein Generator with 3D Structure Prediction</p>
      </div>

      <div style={{ maxWidth: '900px', margin: '40px auto', padding: '0 20px' }}>
        <div style={{ background: 'white', borderRadius: '16px', padding: '40px', boxShadow: '0 20px 60px rgba(0,0,0,0.3)' }}>
          
          <h2 style={{ fontSize: '24px', marginBottom: '30px', color: '#333' }}>Protein Properties</h2>
          
          <PropertySlider name="thermostable" label="🔥 Thermostable" value={properties.thermostable} 
            onChange={(name, val) => setProperties({...properties, [name]: val})} />
          <PropertySlider name="enzyme" label="⚡ Enzyme-like" value={properties.enzyme}
            onChange={(name, val) => setProperties({...properties, [name]: val})} />
          <PropertySlider name="membrane" label="🧱 Membrane" value={properties.membrane}
            onChange={(name, val) => setProperties({...properties, [name]: val})} />
          <PropertySlider name="structural" label="🏗️ Structural" value={properties.structural}
            onChange={(name, val) => setProperties({...properties, [name]: val})} />
          <PropertySlider name="flexible" label="🌊 Flexible" value={properties.flexible}
            onChange={(name, val) => setProperties({...properties, [name]: val})} />

          <div style={{ borderTop: '2px solid #eee', paddingTop: '30px', marginTop: '30px' }}>
            <h2 style={{ fontSize: '20px', marginBottom: '20px', color: '#333' }}>Generation Settings</h2>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#555' }}>
                Number of Sequences: {numSequences}
              </label>
              <input type="range" min="1" max="10" value={numSequences}
                onChange={(e) => setNumSequences(parseInt(e.target.value))}
                style={{ width: '100%', height: '6px', borderRadius: '5px', background: '#ddd', cursor: 'pointer', accentColor: '#667eea' }} />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#555' }}>
                Length: {maxLength} AA
              </label>
              <input type="range" min="20" max="150" value={maxLength}
                onChange={(e) => setMaxLength(parseInt(e.target.value))}
                style={{ width: '100%', height: '6px', borderRadius: '5px', background: '#ddd', cursor: 'pointer', accentColor: '#764ba2' }} />
              <small style={{ fontSize: '11px', color: '#999' }}>Keep under 150 for faster folding</small>
            </div>

            <div style={{ marginBottom: '30px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', color: '#555' }}>
                Temperature: {temperature.toFixed(1)}
              </label>
              <input type="range" min="0.1" max="2.0" step="0.1" value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                style={{ width: '100%', height: '6px', borderRadius: '5px', background: '#ddd', cursor: 'pointer', accentColor: '#764ba2' }} />
            </div>

            <button onClick={handleGenerate} disabled={loading}
              style={{
                width: '100%', padding: '16px', fontSize: '18px', fontWeight: '600',
                background: loading ? '#ccc' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white', border: 'none', borderRadius: '8px',
                cursor: loading ? 'not-allowed' : 'pointer', transition: 'transform 0.2s',
                marginBottom: '10px'
              }}
              onMouseEnter={(e) => !loading && (e.target.style.transform = 'scale(1.02)')}
              onMouseLeave={(e) => !loading && (e.target.style.transform = 'scale(1)')}>
              {loading ? '🔄 Generating...' : '⚡ Generate Proteins'}
            </button>

            {generatedSequences.length > 0 && (
              <button onClick={handleExport}
                style={{
                  width: '100%', padding: '14px', fontSize: '16px', fontWeight: '600',
                  background: '#28a745', color: 'white', border: 'none', borderRadius: '8px',
                  cursor: 'pointer', transition: 'transform 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.02)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}>
                💾 Export as FASTA
              </button>
            )}
          </div>

          {generatedSequences.length > 0 && (
            <div style={{ borderTop: '2px solid #eee', paddingTop: '40px', marginTop: '40px' }}>
              <h3 style={{ fontSize: '20px', marginBottom: '20px', color: '#333' }}>
                Generated Sequences ({generatedSequences.length})
              </h3>
              {generatedSequences.map((seq, i) => (
                <div key={i} style={{
                  background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                  padding: '20px', borderRadius: '8px', border: '1px solid #ddd',
                  marginBottom: '15px'
                }}>
                  <div style={{ fontSize: '12px', fontWeight: '600', color: '#667eea', marginBottom: '10px' }}>
                    SEQUENCE {i + 1}
                  </div>
                  <div style={{
                    fontFamily: 'monospace', fontSize: '13px', lineHeight: '1.6',
                    color: '#333', wordBreak: 'break-all',
                    background: 'rgba(255,255,255,0.5)', padding: '12px',
                    borderRadius: '4px', overflowX: 'auto', marginBottom: '12px'
                  }}>
                    {seq}
                  </div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '12px' }}>
                    Length: <strong>{seq.length}</strong> amino acids
                  </div>
                  <button 
                    onClick={() => handlePredictStructure(seq, i)}
                    disabled={structureLoading && selectedSequence === i}
                    style={{
                      padding: '10px 16px', fontSize: '14px', fontWeight: '600',
                      background: structureLoading && selectedSequence === i ? '#ccc' : '#ff6b6b',
                      color: 'white', border: 'none', borderRadius: '6px',
                      cursor: structureLoading && selectedSequence === i ? 'not-allowed' : 'pointer'
                    }}>
                    {structureLoading && selectedSequence === i ? '⏳ Predicting...' : '🔬 Predict 3D Structure'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div style={{ background: 'rgba(0,0,0,0.2)', color: 'white', textAlign: 'center', padding: '30px 20px', marginTop: '60px' }}>
        <p style={{ fontSize: '14px', opacity: 0.8 }}>Protogen v3.0 • 3D Structure • ESMFold • Mol* Viewer</p>
      </div>
    </div>
  );
}

export default App;
